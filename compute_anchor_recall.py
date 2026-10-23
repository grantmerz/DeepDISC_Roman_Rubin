"""
compute_anchor_recall.py — per-GT best-anchor IoU and RPN proposal recall.

Measures how well the RPN's anchor grid actually covers the ground-truth boxes,
upstream of any detection threshold. For each GT object on a validation slice:

    best_anchor_iou  = max IoU over every anchor the anchor_generator lays down
    best_prop_iou    = max IoU over the post-NMS RPN proposals for that image

Motivation (docs/ps13_architecture_review.md): swin_ps13_ups_lsst_30k.py keeps the
patch-4 anchor sizes [[8],[16],[32],[64],[128]] but moves the anchor strides to
[13,26,52,104,208]. At level 0 that is an 8px anchor on a 13px stride (anchor =
0.6x stride, so there are holes between anchor centers), versus an 8px anchor on a
4px stride for patch-4 (anchor = 2x stride, heavy overlap). The RPN anchor_matcher
is Matcher(thresholds=[0.3, 0.7], labels=[0,-1,1], allow_low_quality_matches=True),
so a GT whose best anchor is below 0.3 is only ever matched through the forced
low-quality path — trained against an anchor that barely overlaps it.

The headline number is therefore frac(best_anchor_iou < 0.3), binned by mag_i.

The natural baseline is `ups_30k` (swin_ups_lsst_30k.py): same 512px upsampled
frame, same anchor sizes, patch_size=4. GT boxes live in the identical pixel frame
as ps13's, so box sizes compare directly and the only variable is patch size.

`base_30k` (swin_lsst_30k.py) is the native-resolution Rubin baseline: same anchor
sizes and strides as ups_30k, but on the native 0.2"/px grid (~148px cutouts padded
to 160). Anchors are fixed in pixels, so the same 128px anchor spans 25.6" here and
7.4" on the upsampled grid. Compare it to ups_30k by `gt_size_arcsec`, not `gt_size_px`.

Example:
    python compute_anchor_recall.py --model-type ups_30k_ps13 --num-images 20
    python compute_anchor_recall.py --model-type ups_30k_ps13
    python compute_anchor_recall.py --model-type ups_30k
    python compute_anchor_recall.py --model-type base_30k
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import torch

# allows us to import from the custom configs directory w/o affecting deepdisc library imports
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs'))
sys.path.insert(0, CONFIG_DIR)

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.events import EventStorage

from custom.mappers import ResizeTestMapper, return_custom_test_loader
from deepdisc.model.loaders import DictMapper

import blend_metrics as bm

DATA_ROOT_DIR = os.path.expanduser('~/lsst_data/')
ANNS_FOLDER = 'annotations_lvl5'
ROOT_RUN_DIR = os.path.expanduser('~/lsst_runs/')

# Upsampled entries use ResizeTestMapper (mirrors ~/run_inference.py's MAPPER_REGISTRY);
# the native baseline uses DictMapper, as in its config.
# ps13 and ups differ only in patch_size -> anchor strides, which is the whole point.
# pixel_scale is arcsec per pixel of the frame the GT boxes live in. The upsampled
# frame is the Roman grid (Lanczos remap by 0.2/0.0575 = 3.478).
RUBIN_PIXEL_SCALE, ROMAN_PIXEL_SCALE = 0.2, 0.0575
MODEL_DEFAULTS = {
    "ups_30k_ps13": {
        "cfgfile":  os.path.join(CONFIG_DIR, "solo/swin_ps13_ups_lsst_30k.py"),
        "run_name": "lsst5_ps13_ups_30k_4h200_bs192_ep50",
        "resize": True,
        "pixel_scale": ROMAN_PIXEL_SCALE,
    },
    "ups_30k": {
        "cfgfile":  os.path.join(CONFIG_DIR, "solo/swin_ups_lsst_30k.py"),
        "run_name": "lsst5_ups_30k_4h200_bs192_ep50",
        "resize": True,
        "pixel_scale": ROMAN_PIXEL_SCALE,
    },
    "base_30k": {
        "cfgfile":  os.path.join(CONFIG_DIR, "solo/swin_lsst_30k.py"),
        "run_name": "lsst5_30k_4h200_bs192_ep50",
        "resize": False,
        "pixel_scale": RUBIN_PIXEL_SCALE,
    },
}
EVAL_DATA_FN = f'{DATA_ROOT_DIR}{ANNS_FOLDER}/val_4k_keypoints_wcs.json'

# RPN anchor_matcher thresholds, from configs/common/models/mask_rcnn_fpn.py.
# Below NEG a GT is only matched via allow_low_quality_matches; above POS it is a
# genuine positive.
MATCHER_NEG, MATCHER_POS = 0.3, 0.7
RECALL_IOUS = (0.5, 0.7)


# ============================================================================
# Setup
# ============================================================================
def load_slice(data_fn, num_images):
    """Load the annotation JSON, truncate to num_images, and build obj_id -> truth cols.

    Returns (dataset_dicts, objid_meta). Magnitudes are joined by obj_id rather
    than by row order: ResizeMapper calls filter_empty_instances() after building
    its annotation list, so `instances` and `annotations` are not index-aligned.
    """
    with open(data_fn) as f:
        dataset_dicts = json.load(f)
    if num_images is not None:
        dataset_dicts = dataset_dicts[:num_images]

    objid_meta = {}
    for d in dataset_dicts:
        for a in d['annotations']:
            objid_meta[a['obj_id']] = (a.get(bm.MAG_COL, np.nan), a.get('size_true', np.nan))
    return dataset_dicts, objid_meta


def build_loader(cfg, dataset_dicts, dataset_name, batch_size, num_workers, resize=True):
    """Register the truncated slice and return a test loader carrying GT instances."""
    for cat in (DatasetCatalog, MetadataCatalog):
        try:
            cat.remove(dataset_name)
        except KeyError:
            pass
    DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
    MetadataCatalog.get(dataset_name).set(thing_classes=list(cfg.metadata.classes))

    cfg.DATASETS.TEST = dataset_name
    cfg.dataloader.test.total_batch_size = batch_size
    cfg.dataloader.test.num_workers = num_workers
    cfg.dataloader.test.persistent_workers = False

    # ResizeTestMapper needs cache_dir + keypoint_hflip_indices, which the standard
    # (DictMapper) branch in run_inference.py does not pass.
    if resize:
        mapper = ResizeTestMapper(
            cfg.dataloader.test.imagereader,
            cfg.dataloader.key_mapper,
            cfg.dataloader.augs,
            keypoint_hflip_indices=cfg.dataloader.test.keypoint_hflip_indices,
            cache_dir=cfg.dataloader.test.cache_dir,
        ).map_data
    else:
        mapper = DictMapper(
            cfg.dataloader.test.imagereader,
            cfg.dataloader.key_mapper,
            cfg.dataloader.augs,
            keypoint_hflip_indices=cfg.dataloader.test.keypoint_hflip_indices,
        ).map_data
    return return_custom_test_loader(cfg, mapper)


def build_model(cfg, ckpt_path, device):
    model = instantiate(cfg.model)
    model.to(device)
    # No create_ddp_model: single process, so attributes are on `model` directly.
    DetectionCheckpointer(model).load(ckpt_path)
    model.eval()
    return model


# ============================================================================
# Measurement
# ============================================================================
def anchor_level_index(anchors_per_level):
    """Flat anchor index -> pyramid level index, as an int64 tensor."""
    counts = [len(a) for a in anchors_per_level]
    return torch.repeat_interleave(
        torch.arange(len(counts)), torch.tensor(counts)
    )


def measure(model, loader, objid_meta, device, verbose=True):
    """One row per GT object. Returns (DataFrame, canvas_shape)."""
    rpn = model.proposal_generator
    rows = []
    canvas = None
    n_batches = len(loader)

    with EventStorage(), torch.no_grad():
        for bi, batch in enumerate(loader):
            images = model.preprocess_image(batch)
            if canvas is None:
                canvas = tuple(images.tensor.shape[-2:])
            features = model.backbone(images.tensor)

            anch_feats = [features[f] for f in rpn.in_features]
            anchors_per_level = rpn.anchor_generator(anch_feats)
            all_anchors = Boxes.cat(anchors_per_level)
            lvl_of = anchor_level_index(anchors_per_level).to(device)

            # gt_instances=None takes the inference path (no losses, post_nms_topk[1])
            proposals, _ = rpn(images, features, None)

            for d, props in zip(batch, proposals):
                gt = d['instances']
                if len(gt) == 0:
                    continue
                gt_boxes = gt.gt_boxes.to(device)

                a_iou, a_idx = pairwise_iou(all_anchors, gt_boxes).max(dim=0)
                if len(props) > 0:
                    p_iou = pairwise_iou(props.proposal_boxes, gt_boxes).max(dim=0).values
                else:
                    p_iou = torch.zeros(len(gt), device=device)

                t = gt_boxes.tensor.cpu().numpy()
                gt_w, gt_h = t[:, 2] - t[:, 0], t[:, 3] - t[:, 1]
                objids = gt.gt_objid.cpu().numpy()
                meta = np.array([objid_meta.get(int(o), (np.nan, np.nan)) for o in objids])

                rows.append(pd.DataFrame({
                    'image_id': d['image_id'],
                    'obj_id': objids,
                    'gt_class': gt.gt_classes.cpu().numpy(),
                    'gt_w': gt_w,
                    'gt_h': gt_h,
                    'gt_size_px': np.sqrt(gt_w * gt_h),
                    'best_anchor_iou': a_iou.cpu().numpy(),
                    'best_anchor_level': lvl_of[a_idx].cpu().numpy(),
                    'best_prop_iou': p_iou.cpu().numpy(),
                    'n_props': len(props),
                    bm.MAG_COL: meta[:, 0],
                    'size_true': meta[:, 1],
                }))

            if verbose and ((bi + 1) % max(n_batches // 10, 1) == 0 or bi + 1 == n_batches):
                print(f'  batch {bi + 1}/{n_batches} '
                      f'({sum(len(r) for r in rows):,} GT objects)', flush=True)

    return pd.concat(rows, ignore_index=True), canvas


# ============================================================================
# Reporting
# ============================================================================
def mag_bin_edges(mag_limit='gold', mag_min=18.5, mag_max=27.5, width=0.5):
    """Bin edges anchored so m_cut falls exactly on an edge.

    Same construction as plot_binned_metrics.NotebookArgs.mag_bins, so these
    curves overlay the existing completeness/purity plots without resampling.
    """
    m_cut = bm.MAG_LIMITS[mag_limit]
    n_below = np.ceil((m_cut - mag_min) / width)
    lo = m_cut - n_below * width
    return np.arange(lo, mag_max + width / 2, width)


def summarize(df):
    """Scalar summary of one run's per-GT table."""
    a, p = df['best_anchor_iou'], df['best_prop_iou']
    s = {
        'N_gt': len(df),
        'anchor_iou_median': a.median(),
        'anchor_iou_p10': a.quantile(0.10),
        'anchor_iou_p90': a.quantile(0.90),
        f'frac_anchor_lt_{MATCHER_NEG}': (a < MATCHER_NEG).mean(),
        f'frac_anchor_ge_{MATCHER_POS}': (a >= MATCHER_POS).mean(),
        'prop_iou_median': p.median(),
    }
    for thr in RECALL_IOUS:
        s[f'rpn_recall@{thr}'] = (p >= thr).mean()
    return s


def binned_table(df, edges):
    """Per-mag-bin stats. Returns a DataFrame indexed by bin center."""
    idx = np.digitize(df[bm.MAG_COL].to_numpy(), edges) - 1
    out = []
    for b in range(len(edges) - 1):
        sub = df[idx == b]
        if len(sub) == 0:
            continue
        row = {'mag_lo': edges[b], 'mag_hi': edges[b + 1],
               'mag_center': 0.5 * (edges[b] + edges[b + 1])}
        row.update(summarize(sub))
        out.append(row)
    return pd.DataFrame(out)


def print_report(model_type, run_name, canvas, cfg_strides, cfg_sizes, df, edges):
    print(f'\n### Anchor / RPN recall — {model_type} ({run_name})\n')
    print(f'- padded canvas (from `preprocess_image`): **{canvas[0]}x{canvas[1]}**')
    print(f'- anchor strides: `{cfg_strides}`')
    print(f'- anchor sizes:   `{cfg_sizes}`')
    print(f'- anchor/stride ratio per level: '
          f'`{[round(s[0] / st, 2) for s, st in zip(cfg_sizes, cfg_strides)]}`')
    print(f'- GT objects: {len(df):,} over {df.image_id.nunique():,} images')
    print(f'- proposals per image: {df.n_props.min()}–{df.n_props.max()}\n')

    s = summarize(df)
    print('| Statistic | Value |')
    print('|---|---|')
    for k, v in s.items():
        print(f'| {k} | ' + (f'{v:,}' if k == 'N_gt' else f'{v:.4f}') + ' |')

    lvl = df.best_anchor_level.value_counts().sort_index()
    print('\n**Best-anchor pyramid level** (fraction of GTs)\n')
    print('| level | ' + ' | '.join(str(i) for i in lvl.index) + ' |')
    print('|' + '---|' * (len(lvl) + 1))
    print('| frac | ' + ' | '.join(f'{c / len(df):.4f}' for c in lvl.values) + ' |')

    bt = binned_table(df, edges)
    print(f'\n**Binned by {bm.MAG_COL}**\n')
    cols = ['N_gt', 'anchor_iou_median', f'frac_anchor_lt_{MATCHER_NEG}',
            f'frac_anchor_ge_{MATCHER_POS}'] + [f'rpn_recall@{t}' for t in RECALL_IOUS]
    print('| mag bin | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (len(cols) + 1))
    for _, r in bt.iterrows():
        cells = [f'{int(r.N_gt):,}'] + [f'{r[c]:.4f}' for c in cols[1:]]
        print(f'| {r.mag_lo:.1f}–{r.mag_hi:.1f} | ' + ' | '.join(cells) + ' |')


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-type', required=True, choices=list(MODEL_DEFAULTS),
                   help='Which run to measure')
    p.add_argument('--cfgfile', default=None, help='Override the config path')
    p.add_argument('--run-name', default=None, help='Override the run directory name')
    p.add_argument('--data-fn', default=EVAL_DATA_FN, help='Annotation JSON to slice')
    p.add_argument('--num-images', type=int, default=500,
                   help='Validation images to use; 0 or negative means all')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--output-dir', default=None,
                   help='Default: {run_dir}/metrics/anchor_recall')
    return p.parse_args()


def main():
    a = parse_args()
    d = MODEL_DEFAULTS[a.model_type]
    cfgfile = a.cfgfile or d['cfgfile']
    run_name = a.run_name or d['run_name']
    run_dir = os.path.join(ROOT_RUN_DIR, run_name)
    num_images = a.num_images if a.num_images and a.num_images > 0 else None
    out_dir = a.output_dir or os.path.join(run_dir, 'metrics', 'anchor_recall')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 80)
    print('ANCHOR IoU / RPN RECALL')
    print('=' * 80)
    for k, v in [('model_type', a.model_type), ('cfgfile', cfgfile),
                 ('run_name', run_name), ('data_fn', a.data_fn),
                 ('num_images', num_images or 'all'), ('device', device),
                 ('output_dir', out_dir)]:
        print(f'  {k}: {v}')

    cfg = LazyConfig.load(cfgfile)
    for key in cfg.get('MISC', dict()).keys():
        cfg[key] = cfg.MISC[key]
    cfg.dataloader.augs = None  # geometry measurement: no augmentation
    ckpt = os.path.join(run_dir, f'{run_name}.pth')
    cfg.train.init_checkpoint = ckpt

    ag = cfg.model.proposal_generator.anchor_generator
    cfg_strides, cfg_sizes = list(ag.strides), [list(s) for s in ag.sizes]
    print(f'  checkpoint: {ckpt}')
    print(f'  anchor strides: {cfg_strides}')
    print(f'  anchor sizes:   {cfg_sizes}')

    print('\nLoading annotations...')
    dataset_dicts, objid_meta = load_slice(a.data_fn, num_images)
    print(f'  {len(dataset_dicts):,} images, {len(objid_meta):,} unique obj_ids')

    loader = build_loader(cfg, dataset_dicts, f'anchor_recall_{a.model_type}',
                          a.batch_size, a.num_workers, resize=d['resize'])
    print('Building model...')
    model = build_model(cfg, ckpt, device)
    assert list(model.proposal_generator.anchor_generator.strides) == cfg_strides, \
        'instantiated anchor strides differ from the config'

    print('\nMeasuring...')
    df, canvas = measure(model, loader, objid_meta, device)
    df.insert(0, 'run', run_name)
    df.insert(0, 'model_type', a.model_type)
    # common physical size so native and upsampled frames compare directly
    df['gt_size_arcsec'] = df['gt_size_px'] * d['pixel_scale']

    if canvas != (cfg.model.backbone.square_pad,) * 2:
        print(f'\n  NOTE: padded canvas is {canvas[0]}x{canvas[1]}, but square_pad is '
              f'{cfg.model.backbone.square_pad}. ImageList.from_tensors rounds the '
              f'square size up to a multiple of backbone.size_divisibility '
              f'({model.backbone.size_divisibility}). Anchor placement uses the '
              f'actual canvas.\n')

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'anchor_recall_{run_name}.csv')
    df.to_csv(csv_path, index=False)

    print_report(a.model_type, run_name, canvas, cfg_strides, cfg_sizes,
                 df, mag_bin_edges())
    print(f'\n_Per-GT CSV → {csv_path}_')


if __name__ == '__main__':
    main()
