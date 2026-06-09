# Common Libraries
"""
Pretty similar to run_inference.py except we're extracting embeddings directly 
from foreground proposals before NMS rather than from the final detections

To do this, we're going to be doing the exact same flow as forward_contrastive()
    RPN -> label_and_sample_proposals -> add_ground_truth_to_proposals
    -> select_foreground_proposals -> ROI-pool -> proj_q / proj_k -> z_q / z_k

and then save per proposal: z_q, z_k, gt_objid, gt_classes, file_name

And the test score and NMS thresholds don't matter here since they only apply during the inference path (_forward_box in eval mode)
which does class scoring --> score filtering --> NMS --> surviving detections.
We're taking the training path which is controlled by completely different params 
(IoU thresholds in proposal matchers, batch_size_per_image, etc) set in config and fixed at model construction time.

So there's exactly one output per checkpoint per data split

Usage:
    python extract_proposal_embeddings.py --run_name clip5_supcon_30k_4h200_bs64_ep15 --num_gpus 4
    python extract_proposal_embeddings.py --run_name clip5_30k_4h200_bs64_ep50 --num_gpus 4

"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import json
import traceback
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gc
import time
import random
import socket
import numpy as np

# allows us to import from the custom configs directory w/o affecting deepdisc library imports
sys.path.insert(0, '/u/yse2/deepdisc/configs')

# Detectron2 setup
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import launch
from detectron2.data import DatasetCatalog, MetadataCatalog
import pycocotools.mask as mask_util
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

# DeepDisc imports
from custom.mappers import CLIPTestWithRomanMapper, return_custom_test_loader
from deepdisc.data_format.register_data import register_data_set


def select_dist_port() -> int:
    """Return a per-run distributed port with collision avoidance.

    Priority:
    1) Respect ``MASTER_PORT`` if provided by the scheduler/launcher.
    2) Derive a stable per-job candidate from ``SLURM_JOB_ID``.
    3) Fall back to a random high port.
    4) If candidate is busy, ask the OS for a free local port.
    """
    master_port = os.environ.get("MASTER_PORT")
    if master_port is not None:
        try:
            port = int(master_port)
            if 1 <= port <= 65535:
                print(f"Using {port}")
                return port
        except ValueError:
            pass

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id and job_id.isdigit():
        candidate = 15000 + (int(job_id) % 45000)
    else:
        candidate = random.SystemRandom().randint(15000, 60000)

    # Probe candidate port on localhost; if unavailable, ask OS for any free port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", candidate))
            return candidate
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

def get_pooler_and_proj(roi_heads):
    """
    Return (pooler, proj_q, proj_k, use_flatten) for whichever ROIHeads variant
    is loaded. Handles both AdaptiveMaxPool and flatten variants
    """
    if hasattr(roi_heads, 'feature_pooler'):
        pooler = roi_heads.feature_pooler
    elif hasattr(roi_heads, 'query_feature_pooler'):
        pooler = roi_heads.query_feature_pooler
    else:
        raise AttributeError("ROIHeads has neither feature_pooler nor query_feature_pooler")
    proj_q = roi_heads.proj_q
    proj_k = roi_heads.proj_k

    # flatten variant has proj_q input dim = 256 * 7 * 7 = 12544
    # AdaptiveMaxPool variant has proj_q input dim = 256
    first_linear = proj_q.net[0]
    use_flatten = (first_linear.in_features > 256)
    return pooler, proj_q, proj_k, use_flatten


def pool_and_proj(pooler, proj, features, boxes, box_in_features, use_flatten, compute_pooled=False, compute_proj=True):
    """
    ROI-pool -> reduce spatial dims -> optional projection / optional L2 norm.

    Returns:
        (proj_out, pooled) where either item can be None depending on flags
    """
    features = [features[f] for f in box_in_features]
    pooled = pooler(features, boxes)  # (N, 256, 7, 7)
    if use_flatten:
        pooled = torch.flatten(pooled, 1)       # (N, 256*7*7)
    else:
        pooled = nn.AdaptiveMaxPool2d((1, 1))(pooled)  # (N, 256, 1, 1)
        pooled = torch.flatten(pooled, 1)       # (N, 256)
    proj_out = proj(pooled) if compute_proj else None
    pooled_norm = F.normalize(pooled, dim=1) if compute_pooled else None
    return proj_out, pooled_norm # (N, contrastive_dim), L2-normalized , (N, 256*7*7) or (N, 256)

def extract(num_gpus, cfg_file, run_name, test_data_fn, data_split, save_repr='z_q'):
    """
    Main extract function to be launched on each GPU
    Will be called by detectron2.engine.launch() for each GPU process.
    Args:
        num_gpus: Number of GPUs to use
        cfg_file: Path to the config file
        run_name: Name of the run directory
        test_data_fn: Path to the test data file
        data_split: Which data split to use: 'eval' or 'test'
        save_repr: Which representation to save: 'both', 'z_q', or 'pooled_q'
    """
    logger = setup_logger()
    save_z = save_repr in ("both", "z_q")
    save_pooled = save_repr in ("both", "pooled_q")
    use_pooled_chunking = save_pooled

    try:
        logger.info(f"Process started on GPU rank {comm.get_local_rank()}/{comm.get_world_size()}")
        base_run_dir = os.path.expanduser('~/lsst_runs/')
        run_dir = f'{base_run_dir}{run_name}'
        cfg_file = os.path.expanduser(cfg_file)
        model_path = f'{run_dir}/{run_name}.pth'
        
        if comm.is_main_process():
            logger.info("-"*45)
            logger.info("Setting up proposal-level embedding extraction...")
            logger.info("-"*45)
            logger.info(f"Number of GPUs: {num_gpus}")
            logger.info(f"World size: {comm.get_world_size()}")
            logger.info(f"Data split: {data_split}")
            logger.info(f"Running with parameters:")
            logger.info(f"  Test Data File: {test_data_fn}")
            logger.info(f"  Run Directory: {run_dir}")
            logger.info(f"  Run Name: {run_name}")
            logger.info(f"  Model Path: {model_path}")
            logger.info(f"  Config File: {cfg_file}")
            logger.info(f"  Save Representation: {save_repr}")
            logger.info(f"  Use pooled chunking path: {use_pooled_chunking}")

        cfg = LazyConfig.load(f"{cfg_file}")
        for key in cfg.get("MISC", dict()).keys():
            cfg[key] = cfg.MISC[key]
        cfg.DATASETS.TEST = "test"
        cfg.dataloader.augs = None  # no augs for test set since we want preds on OG images
        cfg.dataloader.test.mapper = CLIPTestWithRomanMapper
        # model params
        cfg.train.init_checkpoint = model_path
        # register test dataset on ALL processes b/c each GPU needs access to the dataset
        if comm.is_main_process():
            logger.info(f"Registering test dataset from: {test_data_fn}")
        try:
            DatasetCatalog.remove(cfg.DATASETS.TEST)
            MetadataCatalog.remove(cfg.DATASETS.TEST)
        except Exception:
            pass
        custom_colors = [
            (0, 255, 0),    # green for galaxies
            (0, 0, 255),    # blue for stars
        ]
        # need this becasue the test Dataloader works on registered datasets, 
        # so we have to register the test dataset even if there's no visualization
        register_data_set(
            cfg.DATASETS.TEST, test_data_fn, 
            thing_classes=cfg.metadata.classes, thing_colors=custom_colors
        )
        # synch all processes after dataset registration
        comm.synchronize()
        
        # adjust based on num GPUs and GPU memory
        # for H200,
        # for A40, can do bs=8 per GPU w/ 16 workers
        # since no grads, can do higher test batch size compared to train batch size
        # additionally, we'll need to set total batch size to be divisible by num_gpus
        # so that each GPU gets an equal share of the test set
        train_bs = cfg.dataloader.train.total_batch_size
        test_total_bs = 32 * 3 * num_gpus
        cfg.dataloader.test.total_batch_size = test_total_bs // num_gpus  # higher since no grads
        # https://discuss.pytorch.org/t/dataloader-persistent-workers-usage/189329/3
        cfg.dataloader.test.persistent_workers = True  # keep workers alive b/w threshold combos
        cfg.dataloader.test.num_workers = 8  # 16 crashed for bs=96 w/ 4x A100x8
        if comm.is_main_process():
            logger.info(f"Training Batch Size: {train_bs}")
            logger.info(f"Test Batch size across all GPUs: {test_total_bs}")
            logger.info(f"Test Batch size per GPU: {cfg.dataloader.test.total_batch_size}")
        
        imagereader = cfg.dataloader.train.imagereader
        mapper = cfg.dataloader.test.mapper(
            imagereader, cfg.dataloader.key_mapper, cfg.dataloader.augs
        ).map_data
        test_loader = return_custom_test_loader(cfg, mapper)
        if torch.cuda.is_available():
            cfg.train.device = "cuda"
        else:
            cfg.train.device = "cpu"
        if comm.is_main_process():
            logger.info(f"Using device: {cfg.train.device}")
            logger.info("Creating model...")
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        # wraps model in DistributedDataParallel if world_size > 1 so that each GPU will have its own copy of the model
        model = create_ddp_model(model)
        if comm.is_main_process():
            logger.info(f"Model wrapped in DDP: {isinstance(model, torch.nn.parallel.DistributedDataParallel)}")
            logger.info(f"Loading checkpoint from: {cfg.train.init_checkpoint}")
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.train.init_checkpoint)
        # training mode so label_and_sample_props/add_gt_to_props/select_fg_props will run with the training params
        model.train()
        
        mod = model.module if hasattr(model, 'module') else model # DDP has module but non-DDP doesn't
        roi_heads = mod.roi_heads
        pooler, proj_q, proj_k, use_flatten = get_pooler_and_proj(roi_heads)

        if comm.is_main_process():
            logger.info("Checkpoint loaded successfully.")
            logger.info(f"  ROI heads class:  {type(roi_heads).__name__}")
            logger.info(f"  Use flatten:      {use_flatten}")
            logger.info(f"  Proj input dim:   {proj_q.net[0].in_features}")
            logger.info(f"  Contrastive dim:  {proj_q.net[-1].out_features}")
            logger.info(f"  Model in training mode: {model.training}")

        output_dir = f'{run_dir}/preds/{data_split}'
        os.makedirs(output_dir, exist_ok=True)
        emb_output_dir = os.path.join(output_dir, 'pre_nms_embeddings')
        os.makedirs(emb_output_dir, exist_ok=True)

        # For pooled modes, write chunk files directly under run output dir
        # For z-only mode, keep temp-shard behavior
        if use_pooled_chunking:
            shard_dir = os.path.join(emb_output_dir, f'chunk_shards')
        else:
            # /tmp on Delta GPU nodes is 1.5TB local NVMe SSD, unique per node and job
            # using it for shards avoids gloo TCP timeout on large comm.gather() calls
            shard_dir = f'/tmp/shards_{run_name}'
        os.makedirs(shard_dir, exist_ok=True)
        # sync before starting extraction
        comm.synchronize()
        
        batch_extract_times = []
        batch_sizes = []
        per_img_extract_times = []
        per_img_data_times = []
        per_img_preprocess_times = []
        per_img_backbone_feature_times = []
        per_img_rpn_times = []
        per_img_total_times = []
        total_compute_time = 0.0
        
        shard_file_names = []
        shard_z_q = []
        shard_z_k = []
        shard_pooled_q = []
        shard_pooled_k = []
        shard_gt_objids = []
        shard_gt_classes = []
        total_num_props = 0
        rank = comm.get_rank()
        chunk_idx = 0
        chunk_meta = []
        chunk_prop_budget = 250000 if use_pooled_chunking else 450000
        chunk_prop_budget_env = os.environ.get("EMB_CHUNK_PROP_BUDGET")
        if use_pooled_chunking and chunk_prop_budget_env is not None:
        # need to do per-rank chunking and saving to avoid OOM on comm.gather() for large objects (mainly pooled_qs for 900k proposals across all GPUs)
            try:
                chunk_prop_budget = max(1, int(chunk_prop_budget_env))
            except ValueError:
                if comm.is_main_process():
                    logger.warning(
                        f"Invalid EMB_CHUNK_PROP_BUDGET='{chunk_prop_budget_env}', "
                        f"using default {chunk_prop_budget}"
                    )
        if comm.is_main_process() and use_pooled_chunking:
            logger.info(
                f"Chunk proposal budget per rank: {chunk_prop_budget} "
                f"(save_repr={save_repr})"
            )

        def flush_chunk_to_disk():
            """Flush currently buffered proposals for this rank to a chunk file"""
            nonlocal shard_file_names
            nonlocal shard_z_q
            nonlocal shard_z_k
            nonlocal shard_pooled_q
            nonlocal shard_pooled_k
            nonlocal shard_gt_objids
            nonlocal shard_gt_classes
            nonlocal chunk_idx
            nonlocal chunk_meta
            if not shard_gt_objids:
                return 0
            chunk_data = {
                'gt_objid': torch.cat(shard_gt_objids, dim=0),
                'gt_classes': torch.cat(shard_gt_classes, dim=0),
                'file_name': shard_file_names
            }
            props_in_chunk = int(chunk_data['gt_objid'].shape[0])
            chunk_data['num_props'] = props_in_chunk
            if save_z:
                chunk_data['z_q'] = torch.cat(shard_z_q, dim=0) if shard_z_q else torch.empty(0, 128)
            if save_pooled:
                chunk_data['pooled_q'] = torch.cat(shard_pooled_q, dim=0) if shard_pooled_q else torch.empty(0, 256)
            if save_z and shard_z_k:
                chunk_data['z_k'] = torch.cat(shard_z_k, dim=0)
            if save_pooled and shard_pooled_k:
                chunk_data['pooled_k'] = torch.cat(shard_pooled_k, dim=0)
            chunk_path = os.path.join(shard_dir, f'rank{rank}_chunk{chunk_idx:04d}.pt')
            torch.save(chunk_data, chunk_path)

            # Sidecar metadata file used by downstream analytics to avoid loading
            # full embedding chunks when only labels/object ids are needed
            meta_path = chunk_path.replace('.pt', '_meta.pt')
            meta_data = {
                'gt_objid': chunk_data['gt_objid'],
                'gt_classes': chunk_data['gt_classes'],
                'file_name': chunk_data['file_name'],
                'num_props': props_in_chunk,
            }
            torch.save(meta_data, meta_path)
            chunk_meta.append({
                'path': chunk_path,
                'meta_path': meta_path,
                'num_props': props_in_chunk,
            })
            chunk_idx += 1
            shard_file_names = []
            shard_z_q = []
            shard_z_k = []
            shard_pooled_q = []
            shard_pooled_k = []
            shard_gt_objids = []
            shard_gt_classes = []
            return props_in_chunk
        
        total = len(test_loader)
        prev_time = time.perf_counter()  # track time for data loading
        if comm.is_main_process():
            logger.info("-"*45)
            logger.info("Starting extraction...")
            logger.info("-"*45)
            logger.info(f"Total batches per GPU: {total}")
            logger.info(f"Total batches across all GPUs: {total * num_gpus}")
        start_time = time.perf_counter()
        with EventStorage():
            with torch.no_grad():
                for idx, batch in enumerate(test_loader):
                    batch_start = time.perf_counter()
                    batch_size = len(batch)
                    batch_sizes.append(batch_size)
                    # data loading time is time since last batch ended
                    data_load_time = batch_start - prev_time
                    extract_start = time.perf_counter()
                    preprocess_start = time.perf_counter()
                    imgs_q_rubin = mod.preprocess_image(batch, "image_rubin", 
                                                        mod.rubin_pixel_mean, mod.rubin_pixel_std, 
                                                        mod.backbone_q.size_divisibility, 
                                                        mod.backbone_q.padding_constraints)
                    preprocess_time = time.perf_counter() - preprocess_start
                    
                    gt_instances = [x["instances"].to(cfg.train.device) for x in batch]
                    backbone_start = time.perf_counter()
                    features_q = mod.backbone_q(imgs_q_rubin.tensor)
                    features_k = None
                    if "image_roman" in batch[0]:
                        imgs_k_roman = mod.preprocess_image(batch, "image_roman", 
                                                            mod.roman_pixel_mean, mod.roman_pixel_std, 
                                                            mod.backbone_k.size_divisibility, 
                                                            mod.backbone_k.padding_constraints)
                        features_k = mod.backbone_k(imgs_k_roman.tensor)
                    backbone_time = time.perf_counter() - backbone_start
                    # rpn proposals 
                    rpn_start = time.perf_counter()
                    proposals, _ = mod.proposal_generator(imgs_q_rubin, features_q, gt_instances)
                    labeled_proposals = roi_heads.label_and_sample_proposals(proposals, gt_instances)
                    props_with_gt = add_ground_truth_to_proposals(gt_instances, labeled_proposals)
                    fg_instances, _ = select_foreground_proposals(props_with_gt, roi_heads.num_classes)
                    rpn_time = time.perf_counter() - rpn_start
                    
                    n_fg = sum(len(inst) for inst in fg_instances)
                    if n_fg == 0:
                        if comm.is_main_process():
                            logger.warning(f"No foreground proposals in batch {idx+1}/{total} on GPU {comm.get_local_rank()}. Skipping this batch.")
                        prev_time = time.perf_counter()
                        continue  # skip if no foreground proposals in this batch (pretty rare)
                    
                    # ROI-pool and project
                    boxes = [x.proposal_boxes for x in fg_instances]
                    z_q, pooled_q = pool_and_proj(pooler, proj_q, features_q, boxes, 
                                                  roi_heads.in_features, use_flatten, 
                                                  compute_pooled=save_pooled, compute_proj=save_z)
                    z_k = None
                    pooled_k = None
                    if features_k is not None:
                        z_k, pooled_k = pool_and_proj(pooler, proj_k, features_k, boxes, 
                                                      roi_heads.in_features, use_flatten,
                                                      compute_pooled=save_pooled, compute_proj=save_z)
                    
                    # get gt_objid, gt_classes, file_names
                    objids = torch.cat([inst.gt_objid for inst in fg_instances]).cpu()
                    gt_cls = torch.cat([inst.gt_classes for inst in fg_instances]).cpu()
                    for i, inst in enumerate(fg_instances):
                        shard_file_names.extend([batch[i]["file_name"]] * len(inst))
                    if z_q is not None:
                        shard_z_q.append(z_q.cpu())
                    if pooled_q is not None:
                        shard_pooled_q.append(pooled_q.cpu().half())
                    if z_k is not None:
                        shard_z_k.append(z_k.cpu())
                    if pooled_k is not None:
                        shard_pooled_k.append(pooled_k.cpu().half())
                    shard_gt_objids.append(objids)
                    shard_gt_classes.append(gt_cls)
                    total_num_props += n_fg
                    if use_pooled_chunking:
                        buffered_props = int(sum(t.numel() for t in shard_gt_objids))
                        if buffered_props >= chunk_prop_budget:
                            flushed_props = flush_chunk_to_disk()
                            if comm.is_main_process():
                                logger.info(
                                    f"Flushed rank {rank} chunk {chunk_idx-1:04d} "
                                    f"with {flushed_props} proposals"
                                )

                    extract_time = time.perf_counter() - extract_start
                    batch_extract_times.append(extract_time)
                    total_compute_time += extract_time
                    del imgs_q_rubin, features_q, proposals
                    del labeled_proposals, props_with_gt, fg_instances, boxes
                    if z_q is not None:
                        del z_q
                    if pooled_q is not None:
                        del pooled_q
                    if features_k is not None:
                        del imgs_k_roman, features_k
                        if z_k is not None:
                            del z_k
                        if pooled_k is not None:
                            del pooled_k
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    batch_total_time = time.perf_counter() - batch_start
                    # calculate per-image times
                    per_img_extract = extract_time / batch_size
                    per_img_data = data_load_time / batch_size
                    per_img_pre = preprocess_time / batch_size
                    per_img_backbone = backbone_time / batch_size
                    per_img_rpn = rpn_time / batch_size
                    per_img_total = batch_total_time / batch_size

                    per_img_extract_times.append(per_img_extract)
                    per_img_data_times.append(per_img_data)
                    per_img_preprocess_times.append(per_img_pre)
                    per_img_backbone_feature_times.append(per_img_backbone)
                    per_img_rpn_times.append(per_img_rpn)
                    per_img_total_times.append(per_img_total)

                    prev_time = time.perf_counter()
                    
                    if comm.is_main_process():
                        window_size = 10
                        start_idx = max(0, len(per_img_extract_times) - window_size)
                        rolling_avg_extract = np.mean(per_img_extract_times[start_idx:])
                        rolling_avg_data = np.mean(per_img_data_times[start_idx:])
                        rolling_avg_pre = np.mean(per_img_preprocess_times[start_idx:])
                        rolling_avg_backbone = np.mean(per_img_backbone_feature_times[start_idx:])
                        rolling_avg_rpn = np.mean(per_img_rpn_times[start_idx:])
                        rolling_avg_total = np.mean(per_img_total_times[start_idx:])

                        log_interval = max(total // 4, 1)
                        if (idx + 1) % log_interval == 0 or (idx + 1) == total:
                            elapsed = time.perf_counter() - start_time
                            avg = elapsed / (idx + 1)
                            eta = avg * (total - idx - 1)
                            logger.info(
                                f"GPU {comm.get_local_rank()} | Batch {idx+1}/{total} "
                                f"({(idx+1)/total*100:.0f}%) | "
                                f"Images: {batch_size} | Proposals: {total_num_props}"
                            )
                            logger.info(
                                f"  Current: "
                                f"Preprocess={per_img_pre*1000:.1f}ms/img, "
                                f"Backbone={per_img_backbone*1000:.1f}ms/img, "
                                f"RPN={per_img_rpn*1000:.1f}ms/img, "
                                f"Extract={per_img_extract*1000:.1f}ms/img, "
                                f"Data={per_img_data*1000:.1f}ms/img, "
                                f"Total={per_img_total*1000:.1f}ms/img"
                            )
                            logger.info(
                                f"  Rolling (last {min(window_size, len(per_img_extract_times))}): "
                                f"Preprocess={rolling_avg_pre*1000:.1f}ms/img, "
                                f"Backbone={rolling_avg_backbone*1000:.1f}ms/img, "
                                f"RPN={rolling_avg_rpn*1000:.1f}ms/img, "
                                f"Extract={rolling_avg_extract*1000:.1f}ms/img, "
                                f"Data={rolling_avg_data*1000:.1f}ms/img, "
                                f"Total={rolling_avg_total*1000:.1f}ms/img | "
                                f"ETA: {eta:.1f}s"
                            )

        # sync before saving shards
        comm.synchronize()
        total_time = time.perf_counter() - start_time
        avg_extract_time = np.mean(batch_extract_times)
        total_extract_time = np.sum(batch_extract_times)
        data_time = total_time - total_extract_time

        # -------------------------------------------------------
        # Each rank now saves its own shard to /tmp (local NVMe SSD)
        # instead of communicating large objects over TCP via
        # comm.gather(), which caused gloo timeout crashes
        # -------------------------------------------------------
        rank = comm.get_rank()
        world_size = comm.get_world_size()
        shard_path = os.path.join(shard_dir, f'shard_rank{rank}.pt')
        # results from all GPUs to main process
        if comm.is_main_process():
            logger.info("-"*45)
            logger.info("Saving per-rank shards to disk...")
            logger.info("-"*45)

        if use_pooled_chunking:
            flushed_props = flush_chunk_to_disk()
            if comm.is_main_process() and flushed_props > 0:
                logger.info(
                    f"Flushed rank {rank} final chunk {chunk_idx-1:04d} "
                    f"with {flushed_props} proposals"
                )
            shard_data = {
                'chunk_meta': chunk_meta,
                'total_proposals': total_num_props
            }
            torch.save(shard_data, shard_path)
            if comm.is_main_process():
                logger.info(
                    f"Rank {rank} saved shard manifest with {len(chunk_meta)} chunks "
                    f"and {total_num_props} proposals to: {shard_path}"
                )
        else:
            shard_data = {
                'gt_objid': torch.cat(shard_gt_objids, dim=0) if shard_gt_objids else torch.empty(0, dtype=torch.long),
                'gt_classes': torch.cat(shard_gt_classes, dim=0) if shard_gt_classes else torch.empty(0, dtype=torch.long),
                'file_name': shard_file_names,
                'total_proposals': total_num_props
            }
            if save_z:
                shard_data['z_q'] = torch.cat(shard_z_q, dim=0) if shard_z_q else torch.empty(0, 128)
            if save_pooled:
                shard_data['pooled_q'] = torch.cat(shard_pooled_q, dim=0) if shard_pooled_q else torch.empty(0, 256)
            if save_z and shard_z_k:
                shard_data['z_k'] = torch.cat(shard_z_k, dim=0)
            if save_pooled and shard_pooled_k:
                shard_data['pooled_k'] = torch.cat(shard_pooled_k, dim=0)
            torch.save(shard_data, shard_path)
            if comm.is_main_process():
                logger.info(f"Rank {rank} saved shard with {total_num_props} proposals to: {shard_path}")

        # wait for ALL ranks to finish writing before rank 0 reads
        comm.synchronize()

        if comm.is_main_process():
            all_counts = []
            total_images_per_gpu = sum(batch_sizes)
            avg_preprocess_time = np.mean(per_img_preprocess_times)
            avg_backbone_time = np.mean(per_img_backbone_feature_times)
            avg_rpn_time = np.mean(per_img_rpn_times)
            avg_per_img_extract = np.mean(per_img_extract_times)
            avg_per_img_data = np.mean(per_img_data_times)
            avg_per_img_total = np.mean(per_img_total_times)

            if use_pooled_chunking:
                rank_manifests = []
                total_props_all = 0
                for r in range(world_size):
                    sp = os.path.join(shard_dir, f'shard_rank{r}.pt')
                    shard = torch.load(sp, map_location='cpu', weights_only=False)
                    all_counts.append(shard['total_proposals'])
                    total_props_all += shard['total_proposals']
                    rank_manifests.append(
                        {
                            'rank': r,
                            'total_proposals': shard['total_proposals'],
                            'chunk_meta': shard.get('chunk_meta', [])
                        }
                    )
                    os.remove(sp)

                if save_repr == 'pooled_q':
                    emb_filename = 'proposal_emb_pooled_manifest.pt'
                else:
                    emb_filename = 'proposal_emb_both_manifest.pt'
                emb_output_file = os.path.join(emb_output_dir, emb_filename)
                manifest_dict = {
                    'mode': save_repr,
                    'chunked': True,
                    'run_name': run_name,
                    'data_split': data_split,
                    'chunk_dir': shard_dir,
                    'total_proposals': total_props_all,
                    'per_gpu_counts': all_counts,
                    'rank_manifests': rank_manifests
                }
                torch.save(manifest_dict, emb_output_file)

                logger.info("-"*45)
                logger.info(f"Extraction Done for {total_props_all} proposals across all GPUs.")
                logger.info("-"*45)
                logger.info(f"Total time for {total} batches ({total_images_per_gpu} imgs) per GPU: {total_time:.2f}s")
                logger.info(f"  Average extract time per batch: {avg_extract_time:.2f}s")
                logger.info(f"  Total extract time: {total_extract_time:.2f}s")
                logger.info(f"  Total data loading time: {data_time:.2f}s")
                logger.info("-"*45)
                logger.info("Per-Image Timing Statistics:")
                logger.info(f"  Avg preprocess:    {avg_preprocess_time*1000:.1f}ms/img")
                logger.info(f"  Avg backbone:      {avg_backbone_time*1000:.1f}ms/img")
                logger.info(f"  Avg RPN+proposals: {avg_rpn_time*1000:.1f}ms/img")
                logger.info(f"  Avg extract:       {avg_per_img_extract*1000:.1f}ms/img")
                logger.info(f"  Avg data loading:  {avg_per_img_data*1000:.1f}ms/img")
                logger.info(f"  Avg total:         {avg_per_img_total*1000:.1f}ms/img")
                logger.info(f"  Throughput: {1.0/avg_per_img_total:.1f} imgs/sec")
                logger.info("-"*45)
                logger.info("Proposals per GPU:")
                for gpu_id, count in enumerate(all_counts):
                    logger.info(f"  GPU {gpu_id}: {count} proposals")
                logger.info(f"  Total: {sum(all_counts)} proposals")
                logger.info("=" * 60)
                logger.info(f"Saved pooled manifest: {emb_output_file}")
                logger.info(f"  Chunk dir:      {shard_dir}")
                logger.info(f"  Per-GPU counts: {all_counts}")
                logger.info("=" * 60)
            else:
                # Load and merge all the shards (z-only path)
                all_z_q = []
                all_z_k = []
                all_pooled_q = []
                all_pooled_k = []
                all_objids = []
                all_gt_classes = []
                all_fns = []
                has_roman = False

                for r in range(world_size):
                    sp = os.path.join(shard_dir, f'shard_rank{r}.pt')
                    shard = torch.load(sp, map_location='cpu', weights_only=False)
                    if save_z:
                        all_z_q.append(shard['z_q'])
                    if save_pooled:
                        all_pooled_q.append(shard['pooled_q'])
                    all_objids.append(shard['gt_objid'])
                    all_gt_classes.append(shard['gt_classes'])
                    all_fns.extend(shard['file_name'])
                    all_counts.append(shard['total_proposals'])
                    if save_z and "z_k" in shard:
                        all_z_k.append(shard['z_k'])
                        has_roman = True
                    if save_pooled and "pooled_k" in shard:
                        all_pooled_k.append(shard['pooled_k'])
                    os.remove(sp)

                z_q_all = torch.cat(all_z_q, dim=0) if save_z else None
                pooled_q_all = torch.cat(all_pooled_q, dim=0) if save_pooled else None
                objids_all = torch.cat(all_objids, dim=0)
                gt_cls_all = torch.cat(all_gt_classes, dim=0)

                ref_len = len(z_q_all) if z_q_all is not None else len(pooled_q_all)
                assert ref_len == len(all_fns) == len(objids_all), \
                    f"Mismatch: repr={ref_len}, file_names={len(all_fns)}, objids={len(objids_all)}"

                logger.info("-"*45)
                logger.info(f"Extraction Done for {len(all_fns)} proposals across all GPUs.")
                logger.info("-"*45)
                logger.info(f"Total time for {total} batches ({total_images_per_gpu} imgs) per GPU: {total_time:.2f}s")
                logger.info(f"  Average extract time per batch: {avg_extract_time:.2f}s")
                logger.info(f"  Total extract time: {total_extract_time:.2f}s")
                logger.info(f"  Total data loading time: {data_time:.2f}s")
                logger.info("-"*45)
                logger.info("Per-Image Timing Statistics:")
                logger.info(f"  Avg preprocess:    {avg_preprocess_time*1000:.1f}ms/img")
                logger.info(f"  Avg backbone:      {avg_backbone_time*1000:.1f}ms/img")
                logger.info(f"  Avg RPN+proposals: {avg_rpn_time*1000:.1f}ms/img")
                logger.info(f"  Avg extract:       {avg_per_img_extract*1000:.1f}ms/img")
                logger.info(f"  Avg data loading:  {avg_per_img_data*1000:.1f}ms/img")
                logger.info(f"  Avg total:         {avg_per_img_total*1000:.1f}ms/img")
                logger.info(f"  Throughput: {1.0/avg_per_img_total:.1f} imgs/sec")
                logger.info("-"*45)
                logger.info("Proposals per GPU:")
                for gpu_id, count in enumerate(all_counts):
                    logger.info(f"  GPU {gpu_id}: {count} proposals")
                logger.info(f"  Total: {sum(all_counts)} proposals")
                logger.info("-"*45)

                save_dict = {
                    'gt_objid': objids_all,
                    'gt_classes': gt_cls_all,
                    'file_name': all_fns
                }
                if z_q_all is not None:
                    save_dict['z_q'] = z_q_all
                if pooled_q_all is not None:
                    save_dict['pooled_q'] = pooled_q_all
                if save_z and has_roman:
                    save_dict['z_k'] = torch.cat(all_z_k, dim=0)
                if save_pooled and all_pooled_k:
                    save_dict['pooled_k'] = torch.cat(all_pooled_k, dim=0)

                if save_repr == 'z_q':
                    emb_filename = 'proposal_emb_z.pt'
                else:
                    emb_filename = 'proposal_emb_both.pt'
                emb_output_file = os.path.join(emb_output_dir, emb_filename)
                torch.save(save_dict, emb_output_file)

                logger.info("=" * 60)
                logger.info(f"Saved: {emb_output_file}")
                if z_q_all is not None:
                    logger.info(f"  z_q shape:     {z_q_all.shape}")
                if pooled_q_all is not None:
                    logger.info(f"  pooled_q shape:{pooled_q_all.shape}")
                if save_z and has_roman:
                    logger.info(f"  z_k shape:     {save_dict['z_k'].shape}")
                if save_pooled and 'pooled_k' in save_dict:
                    logger.info(f"  pooled_k shape:{save_dict['pooled_k'].shape}")
                logger.info(f"  gt_objid:      {objids_all.shape}")
                logger.info(f"  Unique objids: {objids_all.unique().numel():,}")
                logger.info(f"  Per-GPU counts: {all_counts}")
                logger.info(f"  Total:         {sum(all_counts):,} proposals")
                logger.info("=" * 60)
        # sync all processes before moving to next combination
        comm.synchronize()
        return "Successfully extracted pre-NMS embeddings for all images!"
    except Exception as e:
        logger.error(f"ERROR on GPU rank {comm.get_local_rank()}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def parse_args():
    """Register CLI arguments and return parsed args."""
    parser = argparse.ArgumentParser(
        description="Extract proposal-level embeddings from a CLIP checkpoint (multi-GPU)"
    )
    parser.add_argument("--run_name", type=str, required=True,
                        help="Run directory name under ~/lsst_runs/")
    parser.add_argument("--cfgfile", type=str, default=None,
                        help="Config file path (auto-resolved from RUN_DEFAULTS if not set)")
    parser.add_argument('--data_split', type=str, default='test', choices=['eval', 'test'],
                        help='Which data split to use: eval (val_ files) or test (test_ files)')
    parser.add_argument("--test_data_fn", type=str, default=None,
                        help="Path to annotation JSON (auto-resolved if not set)")
    parser.add_argument("--num_gpus", type=int, default=4, help='Number of GPUs to use')
    parser.add_argument("--save_repr", type=str, default='z_q', choices=['both', 'z_q', 'pooled_q'],
                        help="Which representation to save: both, z_q only, or pooled_q only")
    return parser.parse_args()

if __name__ == "__main__":
    # config params (os.path.expanduser expands the ~ in our path)
    data_root_dir = os.path.expanduser('~/lsst_data/')
    anns_folder = 'annotations_lvl5'
    RUN_DEFAULTS = {
        "clip5_30k_4h200_bs192_ep15": {
            "cfgfile": "/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_30k.py",
        },
        "clip5_30k_4h200_bs192_ep15_lprj": {
            "cfgfile": "/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_30k.py",
        },
        "clip5_30k_4h200_bs64_ep50": {
            "cfgfile": "/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_30k_og.py",
        },
        "clip5_flatten_30k_4h200_bs64_ep15": {
            "cfgfile": "/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_30k_flatten.py",
        },
        "clip5_flatten_30k_4h200_bs64_ep15_resume": {
            "cfgfile": "/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_30k_flatten.py",
        },
    }
    args = parse_args()
    if args.cfgfile is not None:
        cfgfile = args.cfgfile
    elif args.run_name in RUN_DEFAULTS:
        cfgfile = RUN_DEFAULTS[args.run_name]["cfgfile"]
    else:
        raise ValueError(f"No default config for '{args.run_name}'. Pass --cfgfile.")
    
    if args.test_data_fn is not None:
        test_data_fn = args.test_data_fn
    elif args.data_split == 'test':
        test_data_fn = os.path.join(data_root_dir, anns_folder, 'test_8k_keypoints.json')
    else:
        test_data_fn = os.path.join(data_root_dir, anns_folder, 'val_4k_keypoints.json')
        
    print("-"*45)
    print(f"Starting extraction with {args.num_gpus} GPUs...")
    print(f"Run name: {args.run_name}")
    print(f"Config file: {cfgfile}")
    print(f"Data split: {args.data_split}")
    print(f"Test data file: {test_data_fn}")
    print(f"Save representation: {args.save_repr}")
    print("-"*45)
    port = select_dist_port()
    print(f"Using distributed port: {port}")
    default_dist_url = f"tcp://127.0.0.1:{port}"
    # spawn num_gpus processes, one for each GPU
    launch(
        extract,
        num_gpus_per_machine=args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=default_dist_url,
        args=(args.num_gpus, cfgfile, args.run_name,
              test_data_fn, args.data_split, args.save_repr)
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n" + "="*60)
    print(f"Completed extraction successfully!")
    print("="*60)