# Joint analysis of Roman/Rubin data with DeepDISC

Analysis code for the LSST DESC paper *DeepDISC for Joint Rubin-Roman Source Detection, Deblending, and
Classification*. It builds the Rubin and Roman training data from DC2, trains the DeepDISC models (Rubin-only,
the resampling-ablation variants, and Rubin+Roman), runs inference, matches detections to truth, and computes the
metrics and figures in the paper.

**Software Summary Document (DESC software policy):** *link to be added*

**License:** MIT (see [LICENSE](LICENSE)).

## Installation

The code was run on NCSA Delta (Red Hat Enterprise Linux 9.6, NVIDIA A100 and H200 GPUs) with Python 3.10.
A conda environment with the following versions is known to work:

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

The model configs and the custom detectron2 components (mappers, image readers, poolers, the patch-size-aware Swin
backbone) are in `configs/`. The LSST data-gathering notebooks in `data_processing/lsst/` additionally need the
LSST Science Pipelines (Butler) to read DC2.

## Data Processing Pipeline

This section describes the complete workflow for processing Roman and LSST data for training DeepDISC models.

The Slurm scripts in `jobs/` were written for NCSA Delta. Before running one, edit its `#SBATCH` account and
partition lines, the conda environment, and the hard-coded paths (`/u/yse2/...`, `/work/hdd/bdsp/...`,
data and output directories) to match your system. Submit them from the repository root
(`sbatch jobs/<script>.sh`): they `cd "$SLURM_SUBMIT_DIR"` and refer to scripts and `configs/solo/` by paths
relative to the repository, and the Python scripts find `configs/` relative to their own location.

### 1. Downloading and Preparing Roman Data

After downloading the FITS files, use `data_processing/roman/prepare_roman_data.py` to prepare the data into the COCO format. This script creates cutouts from the Roman images and also generates annotations based on the segmentation masks provided by Troxel that were done by SExtractor. These annotations, as of 10/13/25, aren't used as part of our pipeline yet.

### 2. Checking Roman Cutout Integrity

Before proceeding, verify that none of the Roman cutouts are empty or have NAN values in any of the filters:

```bash
python data_processing/check_cutout_integrity.py
```

This script checks all cutouts (e.g., 157,500 cutouts) and identifies any corrupt data. Example output:
```
Total cutouts checked:           157,500
Corrupt cutouts found:               668 (0.42%)
  - All zero cutouts:                  0
  - Cutouts with NaN values:         668
  - Cutouts with empty bands:          0

Per-band statistics:
  Y106 (band 0):
    - NaN count:                     368
    - Empty count:                     0
  J129 (band 1):
    - NaN count:                     204
    - Empty count:                     0
  H158 (band 2):
    - NaN count:                      96
    - Empty count:                     0
```

The detailed report is saved to a JSON file (e.g., `roman_cutout_integrity_report.json`) containing filenames and bands with corrupt data.

### 3. Obtaining LSST Data

Now, we get all the LSST cutouts and catalogs that overlap with the Roman data. There are four notebooks in `data_processing/lsst/` that handle all the data gathering:

- **LSST_Roman_MasterCatalogs.ipynb**: Creates master LSST truth and detection catalogs
- **LSST_Roman_Butler_Calls.ipynb**: Butler calls to access LSST data
- **LSST_Roman_Cutouts_Catalogs.ipynb**: Creates LSST cutouts and catalogs
- **LSST_Roman_Verify_Cutouts.ipynb**: Verifies LSST cutouts, catalogs and WCS to ensure none of them are empty or contain NAN values.

### 4. Creating LSST Annotations

Create ground truth annotations using Galsim and the truth information:

```bash
# job script for batch processing
sbatch jobs/lsst_anns.sh
# Or run directly with:
python data_processing/lsst_anns.py
```

### 5. Combining and Cleaning Annotations

Once annotations are created, combine all the separate `{tile}.json` COCO-format annotations into one file and clean the data:

```bash
python data_processing/lsst/preprocess_anns.py --root_dir /path/to/lsst_data/ --snr_lvl 5 --integrity_report /path/to/roman_cutout_integrity_report.json
```

This script performs three key steps:
1. **Combines** all separate `{tile}.json` files into `all_metadata.json`
2. **Cleans** the data by removing entries associated with corrupt Roman cutouts (based on the integrity report), creating `all_metadata_clean.json`
3. **Splits** the clean data into train, validation, and test sets (`train.json`, `val.json`, `test.json`)

### 6. Calculating Pixel Statistics

Calculate the mean and standard deviation of all images in the training set (and optionally the test set):

```bash
python data_processing/calc_pixel_mean_std.py --root_dir /u/yse2/lsst_data/ --anns_dir annotations --snr_lvl 5 --num_channels 6 --max_workers 16
```

Example output for 109,782 training images:
```
model.pixel_mean = [
    0.0570717453956604,
    0.05500221252441406,
    0.07863432914018631,
    0.11082269251346588,
    0.13925790786743164,
    0.21512146294116974,
]

model.pixel_std = [
    0.9746726155281067,
    0.6917527318000793,
    0.9822555184364319,
    1.382053017616272,
    1.8204920291900635,
    2.6324615478515625,
]
```

### 7. Training the Model

Each model in the paper has its own config in `configs/solo/`. Set the pixel mean and standard deviation from
step 6, the batch size and the number of epochs there.

| Model in the paper | Config (inference) | Trained with |
| --- | --- | --- |
| DeepDISC Rubin (baseline) | `swin_lsst_30k.py` | `training/run_model_job_30k.py` + `swin_lsst_keypoints.sh`* |
| DeepDISC RubinUps (Lanczos upsampled) | `swin_ups_lsst_30k.py` | `training/train_model_30k.py` |
| DeepDISC RubinPadded | `swin_lsst_30k_pad.py` | `training/train_model_30k.py` |
| DeepDISC PatchSize13 | `swin_ps13_ups_lsst_30k.py` | `training/train_model_30k.py` |
| DeepDISC Rubin+Roman | `swin_comb_lsst_roman_30k.py` | `training/run_model_job_30k_comb.py` with `swin_comb_job_keypoint_30k.py`* |
| Distillation (Future directions) | `swin_distill_job_keypoint_30k.py` | `training/run_model_job_30k_distill.py` + `swin_distill_keypoints.sh`* |
| CLIP (Future directions) | `swin_clip_lsst_roman_30k.py` | `training/train_clip_model_30k.py` + `jobs/train_clip_30k.sh` |

\* In `/work/hdd/bdsp/g4merz/RubinRoman/` on NCSA Delta. `swin_lsst_keypoints.sh` has since been reused for
another run; the baseline was Slurm job 15658778 from that folder. The RubinUps, PatchSize13 and Rubin+Roman
models also need the cached WCS remap maps from `data_processing/cache_wcs_maps.py` / `jobs/run_cache_wcs_maps.sh`.

The RubinUps, RubinPadded and PatchSize13 runs were resumed from intermediate checkpoints, with the resume
epoch and learning rate edited in the training script before each resume. Train with `training/train_model_30k.py`:

```bash
python training/train_model_30k.py --cfgfile configs/solo/swin_lsst_30k.py \
    --train-metadata /path/to/lsst_data/annotations_lvl5/train_30k_keypoints.json \
    --eval-metadata /path/to/lsst_data/annotations_lvl5/val_4k_keypoints.json \
    --num-gpus 4 --run-name lsst5_30k_4h200_bs192_ep50 --output-dir ./lsst_runs/lsst5_30k_4h200_bs192_ep50
```

### 8. Inference over the threshold grid

Run inference on the validation set for every combination of the ROI score threshold and the NMS threshold
(0.20–0.90 and 0.25–0.75 in steps of 0.05; `jobs/run_inference.sh`), then add PSF magnitudes to the
detections:

```bash
python run_inference.py --model_type standard_30k --data_split eval --score_thresholds 0.2 0.25 ... 0.9 \
    --nms_thresholds 0.25 0.3 ... 0.75 --num_gpus 4
python add_dd_mags.py --run_name lsst5_30k_4h200_bs192_ep50 --preds_dir eval
```

### 9. Matching and operating-point selection

`grid_search.py` runs the two-stage FOF + Hungarian matching (`fof_classify.py`) and computes the metrics
(`blend_metrics.py`) for every grid point. `run_select_thresh.sh` picks the F1-optimal operating point and
`run_plot_gs.sh` plots the grid:

```bash
python grid_search.py --run-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 --test-cats-dir /path/to/test_cats \
    --output ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf2/gs_metrics.csv --mag-limit gold --buffer 2
./run_select_thresh.sh lsst30k gold
./run_plot_gs.sh lsst30k gold
```

### 10. Test-set evaluation

Run inference on the test set at the selected operating point (`run_inference.py --data_split test`), add
magnitudes, and build the matched analysis catalogs:

```bash
python run_fof_classify.py --run-name lsst5_30k_4h200_bs192_ep50 --preds-dir test --test-cats-dir /path/to/test_cats \
    --score-thresh 0.5 --nms-thresh 0.6
```

### 11. Paper tables and figures

| Paper content | Code |
| --- | --- |
| Scalar metric tables and confidence intervals | `compute_scalar_table.py`, `blend_metrics.py`, `paper_figures/ci_readme_numbers.py` |
| Magnitude-binned curves | `paper_figures/metrics.ipynb` (uses `plot_binned_metrics.py`) |
| Resampling ablation curves (appendix) | `paper_figures/ablation_metrics.ipynb` |
| Anchor coverage (Section 4.2, appendix) | `compute_anchor_recall.py`, `jobs/anchor_recall.sh`, `paper_figures/anchor_recall.ipynb` |
| Matching-procedure figure | `paper_figures/matching_diagram.py` |
| Attention-window figure | `paper_figures/window_diagram.py` |
| Scene figure | `paper_figures/scene_plots.ipynb` |
| Blend-system examples | `paper_figures/blend_examples.ipynb` |
| Grid-search heatmaps and trade-off plot | `plot_grid_search.py`, `select_thresh.py` |

## File Structure

### Pipeline scripts

| File | Purpose |
| --- | --- |
| `data_processing/lsst_anns.py` | Generates the training labels from GalSim renderings of the truth catalog (step 4) |
| `data_processing/add_keypoints.py` | Adds each object's centroid keypoint to the annotations |
| `training/train_model_30k.py` | Trains a model from a config in `configs/solo/` (step 7) |
| `run_inference.py` | Runs inference for a list of ROI score and NMS thresholds (steps 8 and 10) |
| `add_dd_mags.py` | Adds PSF magnitudes to DeepDISC detections |
| `fof_classify.py` | Two-stage FOF + Hungarian matching and the Stage 1/2 classification |
| `run_fof_classify.py` | Matches one operating point and writes the analysis catalogs (step 10) |
| `blend_metrics.py` | Metric definitions, counts, confidence intervals and magnitude-binned curves for the paper |
| `grid_search.py` | Matching and metrics for every point of the threshold grid (step 9) |
| `select_thresh.py`, `run_select_thresh.sh` | Operating-point selection from the grid-search output |
| `plot_grid_search.py`, `run_plot_gs.sh` | Grid-search heatmaps and purity-vs-completeness plots |
| `compute_scalar_table.py` | Scalar metric tables with confidence intervals |
| `plot_binned_metrics.py` | Magnitude-binned metric curves |
| `compute_anchor_recall.py` | Best-anchor IoU and RPN recall per truth object (anchor-coverage analysis) |
| `paper_figures/matching_diagram.py`, `paper_figures/window_diagram.py` | The matching-procedure and attention-window figures |
| `tests/test_fof_classify.py`, `tests/test_select_thresh.py` | Tests for the matching and the operating-point selection |
| `training/train_clip_model_30k.py`, `training/train_clip_model_100k.py`, `training/convert_checkpoint.py` | CLIP-style Rubin–Roman experiments (Future directions in the paper) |

### Notebooks

| Notebook | Purpose |
| --- | --- |
| `paper_figures/metrics.ipynb` | Magnitude-binned curves and scalar tables for the paper's main results |
| `paper_figures/ablation_metrics.ipynb` | Magnitude-binned curves for the resampling ablation |
| `paper_figures/anchor_recall.ipynb` | Anchor-coverage analysis |
| `paper_figures/scene_plots.ipynb` | Scene-level detection figure |
| `paper_figures/blend_examples.ipynb` | Blend-system example figures |
| `data_processing/galsim_truth_anns.ipynb` | Notebook version of `data_processing/lsst_anns.py` (may be behind the script) |
| `AllObjsHist.ipynb` | Histograms of the Roman object distributions |

### Folders

| Folder | Contents |
| --- | --- |
| `configs/` | Model configs (`solo/`) and custom detectron2 components (`custom/`) |
| `data_processing/` | Roman and LSST data download, cutouts, catalogs, annotations and annotation clean-up (steps 1–6) |
| `training/` | Training scripts and training-loss plots (step 7) |
| `paper_figures/` | Notebooks and scripts that make the paper's figures and quoted numbers |
| `tests/` | Tests for the matching and the operating-point selection (`python -m pytest tests`) |
| `jobs/` | Slurm scripts for data preparation, training, inference and the anchor-coverage analysis |
| `docs/` | Notes, including a review of the patch-size-13 architecture |
| `old/` | Superseded notebooks and scripts, kept for reference |
