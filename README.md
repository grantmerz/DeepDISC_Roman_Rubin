# Joint analysis of Roman/Rubin data with DeepDISC

Analysis code for the LSST DESC paper *DeepDISC for Joint Rubin-Roman Source Detection, Deblending, and
Classification*. It builds the Rubin and Roman training data from DC2, trains the DeepDISC models (Rubin-only,
the resampling-ablation variants and Rubin+Roman), runs inference, matches detections to truth, and computes the
metrics, tables and figures in the paper.

| | |
| --- | --- |
| DESC project | 378 |
| Internal review | [DESC-PUB-00275](https://confluence.slac.stanford.edu/spaces/LSSTDESC/pages/747609133/DESC-PUB-00275) (DESC members only) |
| Software Summary Document (DESC software policy) | [SSD - Google Sheets](https://docs.google.com/spreadsheets/d/1jmmCFLMJOCjgg5UGx8G_t9BHcEZrFLNG/edit?usp=sharing&ouid=116260603488410058508&rtpof=true&sd=true) |
| License | MIT (see [LICENSE](LICENSE)) |

## Branches

| Branch | Contents |
| --- | --- |
| `main` | The code used for the paper (this README) |
| `dp1` | `main` plus the Rubin DP1 inference work (`dp1_inference.py`, `prep_dp1_for_dd.py`, DP1 notebooks) |
| `old` | Superseded scripts and notebooks, kept for reference (the repository as of 2026-10-23, before the clean-up) |

The other branches (`clip`, `grant`, `kd`, `patch_size13`, `pos_roi`, `ssl_demo`, `yash`) are earlier development
branches.

## Installation

The code was run on NCSA Delta (Red Hat Enterprise Linux 9.6, NVIDIA A100 and H200 GPUs) with Python 3.10. The
full `pip freeze` of that environment is in [`requirements_ddbtknv.txt`](requirements_ddbtknv.txt); the main
versions are:

| Package | Version |
| --- | --- |
| PyTorch / torchvision | 2.9.1 / 0.24.1 (CUDA 12.8 build) |
| detectron2 | 0.6, installed from source at commit `fd27788985af` |
| DeepDISC | the `grantmerz/deepdisc` fork, installed in editable mode |
| numpy, scipy, pandas | 1.26.4, 1.15.3, 2.3.3 |
| astropy, galsim | 6.1.7, 2.7.2 |
| FoFCatalogMatching | 0.1.0 |
| opencv-python, scikit-image, sep | 4.12.0, 0.25.2, 1.4.1 |
| pyarrow, scikit-learn, matplotlib | 22.0.0, 1.7.2, 3.10.7 |
| omegaconf, fvcore | 2.3.0, 0.1.5 |

```bash
conda create -n deepdisc_rr python=3.10
conda activate deepdisc_rr
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install 'git+https://github.com/facebookresearch/detectron2.git@fd27788985af0f4ca800bca563acdb700bb890e2'
git clone https://github.com/grantmerz/deepdisc.git && pip install -e ./deepdisc
pip install numpy==1.26.4 scipy==1.15.3 pandas==2.3.3 astropy==6.1.7 galsim==2.7.2 FoFCatalogMatching==0.1.0 \
    opencv-python==4.12.0.88 scikit-image==0.25.2 sep==1.4.1 pyarrow==22.0.0 scikit-learn==1.7.2 matplotlib==3.10.7
git clone https://github.com/grantmerz/DeepDISC_Roman_Rubin.git
```

The LSST data-gathering notebooks in `data_processing/lsst/` also need the LSST Science Pipelines (Butler) to read
DC2; they were run at NERSC.

### Running the code

- Run everything from the repository root. The scripts find `configs/` and the shared analysis modules relative to
  their own location, so no `PYTHONPATH` changes are needed.
- The Slurm scripts in `jobs/` were written for NCSA Delta. Submit them from the repository root
  (`sbatch jobs/<script>.sh`), since they `cd "$SLURM_SUBMIT_DIR"`. Before running one, edit its `#SBATCH` account and
  partition, the conda environment, and the data and output paths (`/u/yse2/...`, `/work/hdd/...`).
- The configs in `configs/solo/` also contain machine-specific paths: `train.init_checkpoint` and, for the
  upsampled and Rubin+Roman models, the WCS-map `cache_dir`.

## Data preparation

### 1. Roman data

Download the Roman coadds of Troxel et al. (2023) with `data_processing/roman/urls.sh` and the scripts in
`data_processing/download_scripts/` (`jobs/download_coadds.sh` runs the download as a job array). Then cut them into
512×512 cutouts in COCO format:

```bash
python data_processing/roman/prepare_roman_data.py        # or: sbatch jobs/prep_roman_data.sh
```

### 2. Roman cutout integrity

Check that no Roman cutout is empty or has NaN values in any filter:

```bash
python data_processing/check_cutout_integrity.py
```

For the 157,500 cutouts this finds 668 (0.42%) with NaN values (368 in Y106, 204 in J129, 96 in H158). The list is
written to `roman_cutout_integrity_report.json`, which step 5 uses to drop those cutouts.

### 3. LSST data

The matching LSST cutouts and catalogs come from DC2 through the Butler (run at NERSC):

- `data_processing/lsst/LSST_Roman_MasterCatalogs.ipynb`: master LSST truth and detection catalogs
- `data_processing/lsst/LSST_Roman_Butler_Calls.ipynb`: Butler access to DC2
- `data_processing/lsst/LSST_Roman_Cutouts_Catalogs.ipynb` and `data_processing/lsst/lsst_cutouts.py`
  (`run_lsst_cutouts.sh`, `shifter_run_lsst_cutouts.sh`): LSST cutouts and per-cutout catalogs
- `data_processing/lsst/LSST_Roman_Verify_Cutouts.ipynb`: checks the cutouts, catalogs and WCS

### 4. Labels

Generate the ground-truth labels with the matched-filter GalSim procedure described in the paper:

```bash
sbatch jobs/lsst_anns.sh                 # or: python data_processing/lsst_anns.py
```

`data_processing/galsim_truth_anns.ipynb` is the notebook version of the same procedure; it also makes the
annotation-example figures in the paper.

### 5. Combine, clean and split

```bash
python data_processing/lsst/preprocess_anns.py --root_dir /path/to/lsst_data/ --snr_lvl 5 \
    --integrity_report /path/to/roman_cutout_integrity_report.json --seed 42     # or: sbatch jobs/preprocess_anns.sh
```

This combines the per-tile `{tile}.json` files into `all_metadata.json`, removes the cutouts flagged in step 2
(`all_metadata_clean.json`), and splits the result 70/10/20 into `train.json`, `val.json` and `test.json`.

The paper uses the first 30,000 images of `train.json` and the same fraction of `val.json` and `test.json`:
`train_30k.json` (30,000 images), `val_4k.json` (4,285) and `test_8k.json` (8,571). `data_processing/make_subsets.py
first-n` makes them:

```bash
python data_processing/make_subsets.py first-n --anns-dir /path/to/lsst_data/annotations_lvl5 --train-size 30000
```

**The paper's split can't be regenerated by re-running this step.** It was made on 2025-10-31 with an unseeded
shuffle, before the `--seed` option existed. The paper's three subsets are listed, in order, in
`data_processing/splits/{train_30k,val_4k,test_8k}_images.csv`. To reproduce the paper, rebuild them from those lists
instead of the first-N cut:

```bash
python data_processing/make_subsets.py from-lists \
    --source /path/to/lsst_data/annotations_lvl5/all_metadata_clean.json --out-dir /path/to/lsst_data/annotations_lvl5
```

### 6. Keypoints and Roman WCS

```bash
python data_processing/add_keypoints.py --root-dir /path/to/lsst_data/ --anns-dir annotations --dset train_30k
#   -> annotations_lvl5/train_30k_keypoints.json
python data_processing/add_roman_wcs.py --root-dir /path/to/lsst_data/ --anns-dir annotations --dset train_30k_keypoints
#   -> annotations_lvl5/train_30k_keypoints_wcs.json
sbatch jobs/run_cache_wcs_maps.sh       # data_processing/cache_wcs_maps.py: Lanczos WCS remap cache
```

Repeat the first two for `val_4k` and `test_8k`. `add_roman_wcs.py` and `cache_wcs_maps.py` read the per-tile
Roman annotations from a hard-coded directory (`/projects/bfhm/yse2/annotations_roman/`); change it to yours. Every model is trained on the `*_keypoints` annotations. The
upsampled models (RubinUps, PatchSize13) and Rubin+Roman need the `*_keypoints_wcs` annotations and the WCS remap
cache.

### 7. Pixel statistics

```bash
python data_processing/calc_pixel_mean_std.py --root_dir /path/to/lsst_data/ --anns_dir annotations --snr_lvl 5 \
    --num_channels 6 --max_workers 16      # or: sbatch jobs/pixel_mean.sh; add --upsample_lsst for upsampled inputs
```

For the 109,782 training images this gives

```
model.pixel_mean = [0.0570717, 0.0550022, 0.0786343, 0.1108227, 0.1392579, 0.2151215]
model.pixel_std  = [0.9746726, 0.6917527, 0.9822555, 1.3820530, 1.8204920, 2.6324615]
```

### 8. Truth and LSST catalogs for matching

`data_processing/test_cats.ipynb` builds the truth and LSST Science Pipelines catalogs for the validation and test
images (`test_cats_lvl5/`), which the matching in steps 11–12 reads with `--test-cats-dir`.

## Training

Each model has its own config in `configs/solo/`. Pretrained weights come from the COCO Swin-B checkpoint
(`cascade_mask_rcnn_swin_b_in21k_model.pkl`); `training/convert_checkpoint.py` converts it for the models that need
a different backbone layout:

```bash
python training/convert_checkpoint.py --input cascade_mask_rcnn_swin_b_in21k_model.pkl \
    --output cascade_mask_rcnn_swin_b_in21k_model_ps13.pkl --single-backbone              # PatchSize13
python training/convert_checkpoint.py --input cascade_mask_rcnn_swin_b_in21k_model.pkl \
    --output cascade_mask_rcnn_swin_b_in21k_clip_roman_rubin_model.pkl                    # CLIP and distillation
```

| Model in the paper | Config | Training script | Slurm job |
| --- | --- | --- | --- |
| Rubin (baseline) | `swin_lsst_30k.py` | `training/run_model_job_30k.py` | — |
| RubinUps (Lanczos upsampled) | `swin_ups_lsst_30k.py` | `training/train_ups_model_30k.py` | `jobs/train_ups_30k.sh` |
| RubinPadded | `swin_lsst_30k_pad.py` | `training/train_pad_model_30k.py` | `jobs/train_30k_pad.sh` |
| PatchSize13 | `swin_ps13_ups_lsst_30k.py` | `training/train_ps13_ups_model_30k.py` | `jobs/train_ps13_ups_30k.sh` |
| Rubin+Roman | `swin_comb_job_keypoint_30k.py` (training), `swin_comb_lsst_roman_30k.py` (inference) | `training/run_model_job_30k_comb.py` | — |
| Distillation (Future directions) | `swin_distill_lsst_roman_30k.py` | `training/run_model_job_30k_distill.py` | `jobs/swin_distill_keypoints.sh` |
| CLIP (Future directions) | `swin_lsst_job_keypoint_clip_roi_30k.py`; `swin_clip_lsst_roman_30k.py` | `training/run_model_job_30k.py`; `training/train_clip_model_30k.py` | `jobs/swin_lsst_clip.sh`; `jobs/train_clip_30k.sh` |

The baseline and Rubin+Roman were trained with job scripts that weren't kept; their commands match the other
training jobs.

For example, RubinUps (from `jobs/train_ups_30k.sh`):

```bash
python training/train_ups_model_30k.py --cfgfile configs/solo/swin_ups_lsst_30k.py \
    --train-metadata /path/to/lsst_data/annotations_lvl5/train_30k_keypoints_wcs.json \
    --eval-metadata /path/to/lsst_data/annotations_lvl5/val_4k_keypoints_wcs.json \
    --num-gpus 4 --run-name lsst5_ups_30k_4h200_bs192_ep50 --output-dir /path/to/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50
```

- **Resumed runs.** RubinUps, RubinPadded and PatchSize13 were resumed from intermediate checkpoints. The resume
  epoch and learning rate are set by the `continue_from_epoch_*` switch near the top of `main()` in each training
  script.
- **Initial weights in the configs.** Each config's `train.init_checkpoint` holds the checkpoint of its last
  resume. To train from scratch, set it to the pretrained checkpoint; in `swin_lsst_30k.py` that path is the
  commented-out line above it.
- **Loss curves.** `training/plot_losses.py` plots the training and validation losses.
- **Other training scripts.** `training/train_model_100k.py` and `training/train_clip_model_100k.py` (with
  `jobs/train_100k.sh` and `jobs/train_clip_100k.sh`) train on the full 110k-image training set, and
  `training/train_model_30k.py` (`jobs/train_30k.sh`) trained an earlier baseline variant. None of these models is
  in the paper.

## Inference, matching and evaluation

### 9. Inference over the threshold grid

Run inference on the validation set for every combination of ROI score threshold (0.20–0.90) and NMS threshold
(0.25–0.75), in steps of 0.05:

```bash
python run_inference.py --model_type standard_30k --data_split eval \
    --score_thresholds 0.2 0.25 ... 0.9 --nms_thresholds 0.25 0.3 ... 0.75 --num_gpus 4     # or: sbatch jobs/run_inference.sh
```

The `--model_type` values are:

| `--model_type` | Model |
| --- | --- |
| `standard_30k` | Baseline |
| `ups_30k` | RubinUps |
| `standard_30k_pad` | RubinPadded |
| `ups_30k_ps13` | PatchSize13 |
| `comb_30k` | Rubin+Roman |
| `distill_30k` | Distillation |
| `clip_30k` | CLIP |

Each type sets its own config and run name.

### 10. Magnitudes

```bash
python add_dd_mags.py --run_name lsst5_30k_4h200_bs192_ep50 --preds_dir eval --filenames_csv /path/to/val4k_fns.csv
```

`--filenames_csv` lists the images inference ran on. For the upsampled and Rubin+Roman runs it also needs their WCS
columns (`*_fns_wcs.csv`). The snippet that builds it from an annotation file is in the comments above the
argument in `add_dd_mags.py`. `jobs/add_dd_mags.sh` and `jobs/submit_add_dd_mags.sh` run it per run.

### 11. Matching and operating-point selection

`grid_search.py` runs the two-stage FOF + Hungarian matching (`fof_classify.py`) and the metrics
(`blend_metrics.py`) for every grid point. `run_select_thresh.sh` (`select_thresh.py`) picks the F1-optimal operating
point, and `run_plot_gs.sh` (`plot_grid_search.py`) plots the grid:

```bash
python grid_search.py --run-dir /path/to/lsst_runs/lsst5_30k_4h200_bs192_ep50 --test-cats-dir /path/to/test_cats_lvl5/val_4k \
    --output /path/to/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf2/gs_metrics.csv --mag-limit gold --buffer 2
./run_select_thresh.sh lsst30k gold         # models: lsst30k | ups30k | comb30k | clip30k
./run_plot_gs.sh lsst30k gold
```

`jobs/gs.sh` runs the grid search for any run directory, and `jobs/grid_search.sh` and `jobs/grid_search_clip.sh`
run it for the paper runs.

### 12. Test-set evaluation

Run inference on the test set at the selected operating point, add magnitudes, and build the matched analysis
catalogs:

```bash
python run_inference.py --model_type standard_30k --data_split test --score_thresholds 0.5 --nms_thresholds 0.6
python add_dd_mags.py --run_name lsst5_30k_4h200_bs192_ep50 --preds_dir test --filenames_csv /path/to/test8k_fns.csv
python run_fof_classify.py --run-name lsst5_30k_4h200_bs192_ep50 --preds-dir test --test-cats-dir /path/to/test_cats_lvl5/test_8k \
    --score-thresh 0.5 --nms-thresh 0.6 --buffer 2 --linking-lengths 1.0 0.5        # or: sbatch jobs/run_fof_classify.sh
```

## Paper tables and figures

| Paper content | Code |
| --- | --- |
| Scalar metric tables and confidence intervals | `compute_scalar_table.py` (uses `blend_metrics.py`) |
| Numbers quoted for the confidence-interval method (DEFF, group sizes) | `paper_figures/ci_readme_numbers.py` |
| Magnitude-binned curves | `paper_figures/metrics.ipynb` (uses `plot_binned_metrics.py`) |
| Resampling-ablation curves (appendix) | `paper_figures/ablation_metrics.ipynb` |
| Anchor coverage and RPN recall (Section 4.2, appendix) | `compute_anchor_recall.py` (`jobs/anchor_recall.sh`), `paper_figures/anchor_recall.ipynb` |
| Grid-search heatmaps and trade-off plot | `plot_grid_search.py`, `select_thresh.py` |
| Matching-procedure figure | `paper_figures/matching_diagram.py` |
| Attention-window figure | `paper_figures/window_diagram.py` |
| Scene figure | `paper_figures/scene_plots.ipynb` |
| Blend-system examples | `paper_figures/blend_examples.ipynb` |
| Annotation examples | `data_processing/galsim_truth_anns.ipynb` |
| Example Rubin and Roman images | `paper_figures/test_moco.ipynb` (paths point at the author's directories) |

## CLIP and distillation (Future directions)

The two strategies mentioned in the paper's Future directions are trained as in the table above. Their embeddings
are analysed with:

- `embed_analysis/extract_pre_nms_embeddings.py` (`jobs/extract_emb.sh`): proposal-level embeddings
- `embed_analysis/coherence.py` (`jobs/plot_emb.sh`): within-object embedding coherence
- `embed_analysis/analyze_embeddings.ipynb`: embedding diagnostics

## Tests

```bash
python blend_metrics.py                 # self-checks of the metric definitions and confidence intervals
python -m pytest tests                  # matching (fof_classify.py) and operating-point selection (select_thresh.py)
```

`tests/test_fof_classify.py` checks the two-stage matching against the paper's catalog-matching section: the
Stage 1 labels, the Hungarian matching and its maximality, the object and blend flags, and the significant-blend
definition. `tests/test_select_thresh.py` checks the metric-direction handling, the cross-linking-length ranking,
the Pareto set, the constraints and the stability check used to choose the operating point.

## Repository layout

| Path | Contents |
| --- | --- |
| `run_inference.py`, `add_dd_mags.py` | Inference over the threshold grid and DeepDISC magnitudes (steps 9–10) |
| `fof_classify.py`, `run_fof_classify.py`, `grid_search.py` | Two-stage matching and the Stage 1/2 classification; one operating point or the whole grid (steps 11–12) |
| `blend_metrics.py`, `plot_binned_metrics.py`, `compute_scalar_table.py` | Metric definitions, confidence intervals, binned curves and scalar tables |
| `select_thresh.py`, `run_select_thresh.sh`, `plot_grid_search.py`, `run_plot_gs.sh` | Operating-point selection and grid plots |
| `compute_anchor_recall.py` | Best-anchor IoU and RPN recall per truth object |
| `configs/` | Model configs (`solo/`), custom detectron2 components (`custom/`: mappers, image readers, poolers, ROI heads, the patch-size-aware Swin backbone) and the COCO templates they build on |
| `data_processing/` | Roman and LSST data, labels, annotation clean-up and subsets, keypoints, WCS and pixel statistics (steps 1–8) |
| `data_processing/splits/` | The paper's `train_30k`, `val_4k` and `test_8k` image lists |
| `training/` | Training scripts, checkpoint conversion and loss plots |
| `paper_figures/` | Notebooks and scripts that make the paper's figures and quoted numbers |
| `embed_analysis/` | Embedding analysis for the CLIP and distillation models |
| `tests/` | Tests for the matching and the operating-point selection |
| `jobs/` | Slurm scripts for every step |
| `assets/` | Example images |
| `requirements_ddbtknv.txt` | Full `pip freeze` of the environment used for the paper |
