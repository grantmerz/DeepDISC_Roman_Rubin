# Common Libraries
import itertools
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import json
import traceback
import torch
import argparse
import gc
import time
import numpy as np
# allows us to import from the custom configs directory w/o affecting deepdisc library imports
# CHANGE TO YOUR OWN CUSTOM PATH TO CONFIGS DIRECTORY
sys.path.insert(0, '/u/yse2/deepdisc/configs')
# Astropy imports
from astropy.wcs import WCS

# Detectron2 setup
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import launch
from detectron2.data import DatasetCatalog, MetadataCatalog
import pycocotools.mask as mask_util
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer

# DeepDisc imports
# COPY CUSTOM MAPPERS CODE FROM DeepDISC Roman Rubin repo to your configs/custom/mappers.py file
from custom.mappers import DP1Mapper
from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import return_test_loader

def inference(num_gpus, cfg_file, model_path, test_data_fn, output_dir,
              topk_per_img, threshold_combos):
    """
    Main inference function to be launched on each GPU.
    Will be called by detectron2.engine.launch() for each GPU process.
    Args:
        num_gpus: Number of GPUs to use
        cfg_file: Path to the config file
        output_dir: Directory to save the inference results
        model_path: Path to the trained model checkpoint
        test_data_fn: Path to the test data file
        topk_per_img: Number of top detections to keep per image
        threshold_combos: List of (score_thresh, nms_thresh) tuples to test
    """
    logger = setup_logger()
    try:
        logger.info(f"Process started on GPU rank {comm.get_local_rank()}/{comm.get_world_size()}")
        # Normalize paths: expand user paths and remove trailing slashes
        cfg_file = os.path.expanduser(cfg_file).rstrip('/')
        model_path = os.path.expanduser(model_path).rstrip('/')
        test_data_fn = os.path.expanduser(test_data_fn).rstrip('/')
        output_dir = os.path.expanduser(output_dir).rstrip('/')
        
        # Create output directory if it doesn't exist (only main process)
        if comm.is_main_process():
            os.makedirs(output_dir, exist_ok=True)
        
        # Sync all processes after directory creation
        comm.synchronize()
        
        if comm.is_main_process():
            logger.info("-"*45)
            logger.info("Setting up inference...")
            logger.info("-"*45)
            logger.info(f"Number of GPUs: {num_gpus}")
            logger.info(f"World size: {comm.get_world_size()}")
            logger.info(f"Running with parameters:")
            logger.info(f"  Test Data File: {test_data_fn}")
            logger.info(f"  Output Directory: {output_dir}")
            logger.info(f"  Model Path: {model_path}")
            logger.info(f"  Config File: {cfg_file}")
            logger.info(f"  topk_per_img: {topk_per_img}")
            logger.info(f"  Total threshold combinations: {len(threshold_combos)}")
            for idx, (s, n) in enumerate(threshold_combos, 1):
                logger.info(f"    {idx}. score={s}, nms={n}")       
        
        cfg = LazyConfig.load(f"{cfg_file}") 
        for key in cfg.get("MISC", dict()).keys():
            cfg[key] = cfg.MISC[key]
        cfg.DATASETS.TEST = "test"
        cfg.dataloader.augs = None # no augs for test set since we want preds on OG images
        cfg.dataloader.test.mapper = DP1Mapper # setting test DataLoader's mapper so that filename gets added to each sample
        # model params
        cfg.train.init_checkpoint = model_path
        # register test dataset on ALL processes b/c each GPU needs access to the dataset
        if comm.is_main_process():
            logger.info(f"Registering test dataset from: {test_data_fn}")
        try:
            DatasetCatalog.remove(cfg.DATASETS.TEST)
            MetadataCatalog.remove(cfg.DATASETS.TEST)
        except:
            pass
        custom_colors = [
            (0, 255, 0),    # green for galaxies
            (0, 0, 255),    # blue for stars
        ]
        astrotest_metadata = register_data_set(
            cfg.DATASETS.TEST, test_data_fn, thing_classes=cfg.metadata.classes, thing_colors=custom_colors
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
        cfg.dataloader.test.total_batch_size = test_total_bs // num_gpus # higher since no grads
        # https://discuss.pytorch.org/t/dataloader-persistent-workers-usage/189329/3
        cfg.dataloader.test.persistent_workers = True # keep workers alive b/w threshold combos
        cfg.dataloader.test.num_workers = 8 # 16 crashed for bs=96 w/ 4x A100x8
        if comm.is_main_process():
            logger.info(f"Training Batch Size: {train_bs}")
            logger.info(f"Test Batch size across all GPUs: {test_total_bs}")
            logger.info(f"Test Batch size per GPU: {cfg.dataloader.test.total_batch_size}")
        mapper = cfg.dataloader.test.mapper(
            cfg.dataloader.imagereader, cfg.dataloader.key_mapper, cfg.dataloader.augs
        ).map_data
        test_loader = return_test_loader(cfg, mapper)
        if torch.cuda.is_available():
            cfg.train.device = "cuda"
        else:
            cfg.train.device = "cpu"
        if comm.is_main_process():
            logger.info(f"Using device: {cfg.train.device}")
            logger.info("Creating model...")
        # changing the model's pixel mean and std from the ones in the config since we're dealing
        # with a different data distribution than the simulated data. We used dp1_pixel_mean_std.py 
        # to get these values. Since these values are used for normalization in the model's forward pass 
        # they can greatly affect the inference results. 
        cfg.model.pixel_mean = [
            2.673048496246338,
            4.275834560394287,
            6.809510707855225,
            6.692336082458496,
            8.969269752502441,
            12.77933406829834,
        ]
        cfg.model.pixel_std = [
            32.0528450012207,
            42.218772888183594,
            70.36144256591797,
            66.37097930908203,
            89.78242492675781,
            145.3740997314453,
        ]
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        
        # wraps model in DistributedDataParallel if world_size > 1 so that each GPU will have its own copy of the model
        model = create_ddp_model(model) 
        if comm.is_main_process():
            logger.info(f"Model wrapped in DDP: {isinstance(model, torch.nn.parallel.DistributedDataParallel)}")
            logger.info(f"Loading checkpoint from: {cfg.train.init_checkpoint}")
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.train.init_checkpoint)
        model.eval()
        if comm.is_main_process():
            logger.info("Checkpoint loaded successfully.")
        # sync before starting inference
        comm.synchronize()
        # loop through every possible combo
        for combo_idx, (test_score_thresh, nms_thresh) in enumerate(threshold_combos, 1):
            if comm.is_main_process():
                logger.info("="*60)
                logger.info(f"Threshold combination {combo_idx}/{len(threshold_combos)}")
                logger.info(f"Score Threshold: {test_score_thresh}, NMS Threshold: {nms_thresh}")
                logger.info("="*60)
            # set thresholds in model
            for box_pred in model.module.roi_heads.box_predictor:
                box_pred.test_topk_per_image = topk_per_img
                box_pred.test_score_thresh = test_score_thresh
                box_pred.test_nms_thresh = nms_thresh
        
            batch_inference_times = []
            batch_sizes = []
            per_img_inference_times = []
            per_img_data_times = []
            per_img_total_times = []
            pred_dicts = []
            file_names = []
            wcs_info = []
            total_compute_time = 0.0
            total_num_dets = 0
            total = len(test_loader)
            prev_time = time.perf_counter()  # track time for data loading
        
            if comm.is_main_process():
                logger.info("-"*45)
                logger.info("Starting inference...")
                logger.info("-"*45)
                logger.info(f"Total batches per GPU: {total}")
                logger.info(f"Total batches across all GPUs: {total * num_gpus}")
        
            start_time = time.perf_counter()
            with torch.no_grad():
                for idx, batch in enumerate(test_loader):
                    batch_start = time.perf_counter()
                    batch_size = len(batch)
                    batch_sizes.append(batch_size)
                    
                    # data loading time is time since last batch ended
                    data_load_time = batch_start - prev_time
                    
                    infer_start = time.perf_counter()
                    # ensure that all CUDA ops are done before starting timing
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    file_names.extend([d['file_name'] for d in batch])
                    wcs_info.extend([d['wcs'] for d in batch])
                    metrics_dict = model(batch)
                    # ensure all CUDA ops complete
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    infer_time = time.perf_counter() - infer_start
                    batch_inference_times.append(infer_time)
                    total_compute_time += infer_time
                    pred_dicts.extend(metrics_dict)
                    total_dets = sum(len(result['instances']) for result in metrics_dict)
                    total_num_dets += total_dets
                    
                    # calculate per-image times
                    per_img_infer = infer_time / batch_size
                    per_img_data = data_load_time / batch_size
                    batch_total_time = time.perf_counter() - batch_start
                    per_img_total = batch_total_time / batch_size
                    
                    per_img_inference_times.append(per_img_infer)
                    per_img_data_times.append(per_img_data)
                    per_img_total_times.append(per_img_total)
                    
                    # calculate rolling averages (last 10 batches)
                    window_size = 10
                    start_idx = max(0, len(per_img_inference_times) - window_size)
                    rolling_avg_infer = np.mean(per_img_inference_times[start_idx:])
                    rolling_avg_data = np.mean(per_img_data_times[start_idx:])
                    rolling_avg_total = np.mean(per_img_total_times[start_idx:])
                    
                    prev_time = time.perf_counter()
                    
                    # Log progress every 25%
                    log_interval = max(total // 4, 1)
                    if (idx + 1) % log_interval == 0 or (idx + 1) == total:
                        if comm.is_main_process():
                            elapsed = time.perf_counter() - start_time
                            avg = elapsed / (idx + 1)
                            eta = avg * (total - idx - 1)
                            logger.info(
                                f"GPU {comm.get_local_rank()} | Batch {idx+1}/{total} ({(idx+1)/total*100:.0f}%) | "
                                f"Images: {batch_size} | Detections: {total_dets}"
                            )
                            logger.info(
                                f"  Current: Infer={per_img_infer*1000:.1f}ms/img, Data={per_img_data*1000:.1f}ms/img, Total={per_img_total*1000:.1f}ms/img"
                            )
                            logger.info(
                                f"  Rolling (last {min(window_size, len(per_img_inference_times))}): "
                                f"Infer={rolling_avg_infer*1000:.1f}ms/img, Data={rolling_avg_data*1000:.1f}ms/img, Total={rolling_avg_total*1000:.1f}ms/img | "
                                f"ETA: {eta:.1f}s"
                            )
        
            # sync before gathering results
            comm.synchronize()
            total_time = time.perf_counter() - start_time
            avg_infer_time = np.mean(batch_inference_times)
            total_infer_time = np.sum(batch_inference_times)
            data_time = total_time - total_infer_time
        
            # let's first move instances to CPU and extract only needed data to avoid GPU OOM during gather
            pred_instances = [d['instances'].to('cpu') for d in pred_dicts]
            # for d in pred_dicts:
            #     instances = d['instances'].to('cpu')
            #     # Extract only what you need, not the full Instance object
            #     pred_dict_cpu = {
            #         'pred_boxes': instances.pred_boxes.tensor.numpy(),
            #         'scores': instances.scores.numpy(),
            #         'pred_classes': instances.pred_classes.numpy(),
            #         'pred_masks': instances.pred_masks.numpy(),
            #     }
            #     pred_dicts_cpu.append(pred_dict_cpu)
            
            # results from all GPUs to main process
            if comm.is_main_process():
                logger.info("-"*45)
                logger.info("Gathering the results from all GPUs...")
                logger.info("-"*45)
            # num of detections from all GPUs
            all_det_counts = comm.gather(total_num_dets, dst=0)
            # all detection dicts and corresponding file names
            gathered_pred_dicts = comm.gather(pred_instances, dst=0)
            gathered_file_names = comm.gather(file_names, dst=0)
            gathered_wcs_info = comm.gather(wcs_info, dst=0)
            # clear GPU memory
            del pred_dicts, pred_instances
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
            if comm.is_main_process():
                # flatten lists
                all_preds = [item for sublist in gathered_pred_dicts for item in sublist]
                all_fns = [item for sublist in gathered_file_names for item in sublist]
                all_wcs = [item for sublist in gathered_wcs_info for item in sublist]
                assert len(all_preds) == len(all_fns) == len(all_wcs), "Mismatch in number of predictions, filenames, or WCS info!!"
                logger.info("-"*45)
                logger.info(f"Inference Done for {len(all_fns)} images.")
                logger.info("-"*45)
                total_images_per_gpu = sum(batch_sizes)
                avg_per_img_infer = np.mean(per_img_inference_times)
                avg_per_img_data = np.mean(per_img_data_times)
                avg_per_img_total = np.mean(per_img_total_times)
                
                logger.info(f"Total time for {total} batches ({total_images_per_gpu} imgs) per GPU: {total_time:.2f}s")
                logger.info(f"  Average inference time per batch: {avg_infer_time:.2f}s")
                logger.info(f"  Total inference time: {total_infer_time:.2f}s")
                logger.info(f"  Total data loading time: {data_time:.2f}s")
                logger.info("-"*45)
                logger.info("Per-Image Timing Statistics:")
                logger.info(f"  Avg inference: {avg_per_img_infer*1000:.1f}ms/img")
                logger.info(f"  Avg data loading: {avg_per_img_data*1000:.1f}ms/img")
                logger.info(f"  Avg total: {avg_per_img_total*1000:.1f}ms/img")
                logger.info(f"  Throughput: {1.0/avg_per_img_total:.1f} imgs/sec")
                logger.info("-"*45)
                logger.info("Detections per GPU:")
                for gpu_id, count in enumerate(all_det_counts):
                    logger.info(f"  GPU {gpu_id}: {count} detections")
                logger.info(f"  Total: {sum(all_det_counts)} detections")
                logger.info("-"*45)
                
                det_ras = []
                det_decs = []
                det_filenames = []
                det_boxes = []
                det_scores = []
                det_classes = []
                det_rle_masks = []
                # now we save all the detections to a detection catalog
                for raw_instances, wcs, fn in zip(all_preds, all_wcs, all_fns):
                    centers_pred = raw_instances.pred_boxes.get_centers().numpy()
                    det_coords = WCS(wcs).pixel_to_world(centers_pred[:,0],centers_pred[:,1])
                    det_ras.extend(det_coords.ra.degree)
                    det_decs.extend(det_coords.dec.degree)
                    pred_boxes = raw_instances.pred_boxes.tensor.numpy()
                    pred_scores = raw_instances.scores.numpy()
                    pred_classes = raw_instances.pred_classes.numpy()
                    pred_masks = raw_instances.pred_masks.numpy()
                    
                    rle_masks = []
                    for mask in pred_masks:
                        # pycocotools expects a Fortran-contiguous array of type uint8
                        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                        # 'counts' field is in bytes, which is not JSON serializable, so we decode it to a string
                        rle['counts'] = rle['counts'].decode('utf-8')
                        rle_masks.append(rle)
                    # rle_masks = mask_util.encode(np.asfortranarray(pred_masks.astype(np.uint8)))
                    # for rle in rle_masks:
                    #     rle['counts'] = rle['counts'].decode('utf-8')
                    num_instances = len(raw_instances)
                    det_filenames.extend(itertools.repeat(fn, num_instances))
                    # det_filenames.extend([fn] * len(raw_instances))
                    det_boxes.extend(pred_boxes.tolist())
                    det_scores.extend(pred_scores.tolist())
                    det_classes.extend(pred_classes.tolist())
                    det_rle_masks.extend(rle_masks)
                logger.info(f"Lengths of det catalog fields: {len(det_ras)}, {len(det_decs)}, {len(det_filenames)}, {len(det_boxes)}, {len(det_scores)}, {len(det_classes)}, {len(det_rle_masks)}")
                assert len(det_ras) == len(det_decs) == len(det_filenames) == len(det_boxes) == len(det_scores) == len(det_classes) == len(det_rle_masks), "Mismatch in lengths of det catalog fields!"
                dd_det_cat = {
                    'id': np.arange(len(det_ras)).tolist(),
                    'ra': det_ras,
                    'dec': det_decs,
                    'class': det_classes, # galaxy=0, star=1
                    'file_name': det_filenames,
                    'bbox': det_boxes,
                    'score': det_scores,
                    'rle_masks': det_rle_masks
                }
                output_file = f'{output_dir}/pred_s{test_score_thresh}_n{nms_thresh}.json'
                with open(output_file, 'w') as f:
                    json.dump(dd_det_cat, f, indent=2)
                # pd.DataFrame(dd_det_cat).to_json(output_file)
                logger.info(f"Detection catalog saved to: {output_file}.")
                logger.info(f"Completed combination {combo_idx}/{len(threshold_combos)}")
                
            # sync all processes before moving to next combination
            comm.synchronize()
            
        return "Successfully completed inference for all threshold combinations."
    except Exception as e:
        logger.error(f"ERROR on GPU rank {comm.get_local_rank()}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # config params (os.path.expanduser expands the ~ in our path)    
    parser = argparse.ArgumentParser(description='Run inference with multiple threshold combinations')
    parser.add_argument('--cfgfile', type=str, required=True, help='Path to the config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--test_data_fn', type=str, required=True, help='Path to the test data file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the inference results')
    parser.add_argument('--topk_per_img', type=int, default=2000,
                        help='Number of top detections to keep per image')
    # confidence score of detections
    parser.add_argument('--score_thresholds', nargs='+', type=float, default=[0.25],
                        help='List of score thresholds to test (e.g., --score_thresholds 0.3 0.4 0.5)')
    # tldr: Non-Maximum Suppression (NMS) threshold allows you to remove duplicate, overlapping bounding boxes from detections
    # More precisely, nms_thresh is the overlap cutoff by Intersection over Union (IoU) used during NMS to decide whether to keep or discard a box.
    # Formally IoU = | Area of Overlap | / | Area of Union | b/w two boxes
    # So high IoU means boxes are overlapping a lot and low IoU means boxes are mostly separate
    # How does NMS work? https://waynestalk.com/en/non-maximum-suppression-nms-en/
    # Step 1) Select the box with highest probability
    # Step 2) Compare the overlap (IoU) of this box with other boxes
    # Step 3) Remove bounding boxes with IoU > iou_threshold (what you set nms_thresh to)
    # Step 4) Then, move to the next box with the highest probability and repeat steps 2-4 until no boxes remain
    # Basically, NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring) box.
    # (For more info: https://detectron2.readthedocs.io/en/latest/modules/layers.html?highlight=nms#detectron2.layers.nms)
    # So if getting too many duplicate dets per obj --> lower nms_thresh meaning you're stricter about removing overlapping boxes
    # If getting too few dets per obj (missing objs) --> raise nms_thresh meaning you're more lenient about keeping overlapping boxes
    parser.add_argument('--nms_thresholds', nargs='+', type=float, default=[0.5],
                        help='List of NMS thresholds to test (e.g., --nms_thresholds 0.2 0.3 0.4)')
    parser.add_argument('--num_gpus', type=int, required=True, help='Number of GPUs to use')
    args = parser.parse_args()
    # https://docs.pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
    # 'file_system' strategy uses file names given to shm_open to identify the shared memory region
    # and avoids caching the file descriptors obtained from it meaning we can have more workers
    torch.multiprocessing.set_sharing_strategy('file_system') # allows for num_workers=16 on test DataLoader
    threshold_combos = list(itertools.product(args.score_thresholds, args.nms_thresholds))
    print("-"*45)
    print(f"Starting inference with {args.num_gpus} GPUs...")
    print(f"Score thresholds: {args.score_thresholds}")
    print(f"NMS thresholds: {args.nms_thresholds}")
    print(f"Total threshold combinations to run: {len(threshold_combos)}")
    print("-"*45)
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    default_dist_url = f"tcp://127.0.0.1:{port}"
    # spawn num_gpus processes, one for each GPU
    launch(
        inference,
        num_gpus_per_machine=args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=default_dist_url,
        args=(args.num_gpus, args.cfgfile, args.model_path, args.test_data_fn, 
              args.output_dir, args.topk_per_img, threshold_combos),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\n" + "="*60)
    print(f"Completed inference with all {len(threshold_combos)} combinations successfully!")
    print("="*60)

# Example usage:
# python dp1_inference.py \
#     --cfgfile /u/yse2/deepdisc/configs/solo/swin_lsst_job.py \
#     --model_path /u/yse2/lsst_runs/lsst5_30k_4h200_bs192_ep50/lsst5_30k_4h200_bs192_ep50.pth \
#     --test_data_fn ~/dp1/test.json \
#     --output_dir ~/dp1/preds \
#     --topk_per_img 2000 \
#     --num_gpus 2 \
#     --score_thresholds 0.25 \
#     --nms_thresholds 0.5

# python dp1_inference.py \ 
# 	--score_thresholds 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 \ 
# 	--nms_thresholds 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75