# Joint analysis of Roman/Rubin data with DeepDISC

**For HAL, OPEN THIS FILE WITH MARKDOWN READER in Jupyter Lab or Preview with Markdown Reader in HAL**

## Data Processing Pipeline

This section describes the complete workflow for processing Roman and LSST data for training DeepDISC models.

### 1. Downloading and Preparing Roman Data

After downloading the FITS files, use `data_processing/roman/prepare_roman_data.py` to prepare the data into the COCO format. This script creates cutouts from the Roman images and generates annotations based on the segmentation masks provided by Troxel that were done by SExtractor.

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

The detailed report is saved to a JSON file (e.g., `/u/yse2/roman_cutout_integrity_report.json`) containing filenames and bands with corrupt data.

### 3. Obtaining LSST Data

Get all the LSST cutouts and catalogs that overlap with the Roman data. There are four notebooks in `data_processing/lsst/` that handle all the data gathering:

- **LSST_Roman_Butler_Calls.ipynb**: Butler calls to access LSST data
- **LSST_Roman_Cutouts_Catalogs.ipynb**: Creates LSST cutouts and catalogs
- **LSST_Roman_MasterCatalogs.ipynb**: Creates master catalogs
- **LSST_Roman_Verify_Cutouts.ipynb**: Verifies LSST cutouts

After obtaining the truth catalogs and images from NERSC and transferring everything to Delta, verify if any of them are empty before proceeding.

### 4. Creating LSST Annotations

Create ground truth annotations using Galsim and the truth information:

```bash
# Use the job script for batch processing
sbatch jobs/lsst_anns.sh

# Or run directly with:
python lsst_anns.py
```

This script creates annotations for objects in cutouts using LSST Truth catalog info with multiprocessing.

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

Modify `swin_lsst.py` to use the ImageNet Swin Transformer by updating:
- Number of epochs based on training set size
- Batch size appropriate for your GPU setup
- Pixel mean and standard deviation values from step 6

Then test your training configuration with:

```bash
python run_model.py --cfgfile ./deepdisc/configs/solo/swin_lsst.py --train-metadata lsst_data/annotations/train.json --eval-metadata lsst_data/annotations/val.json --num-gpus 2 --run-name lsst --output-dir ./lsst_runs/run1
```

## File Structure

This section explains the structure and purpose of the folders/files in this repo.

As of 11/13/24 and based on [LSST/Rubin Project Outline](https://docs.google.com/document/d/1hFqOK-6hv6E2UjG0CJjX5IfqadrR0yOK_ekBkm2A1ns/edit?pli=1&tab=t.0#heading=h.kqvlnv4vmq2p), the most relevant notebooks and scripts are `metrics and metrics_v2` notebooks and `lsst_anns.py`.

### Scripts

**lsst_anns.py**: Creates annotations for objs in cutouts using LSST Truth catalog info (has multiprocessing)

**run_model.py**: Training script.

### Notebooks

**metrics_v2.ipynb (Currently working on as of 11/13/24)**: Slightly improved but incomplete version of **metrics.ipynb**: Using a class instead to set thresholds and make DeepDISC predictions, Plots of Obj Properties, Calcualting Detection Completeness, 1-1 Catalog Matching to Truth Catalog, 2D Histogram of FOF Plots

**metrics.ipynb (Currently working on as of 11/13/24)**: Creating DeepDISC detection catalog, LSST Truth/Detection catalog based on Test Set, Calculating Detection Completeness, 1-1 Catalog Matching to Truth Catalog, Creating 2D Histogram FOF Plots, Creating Unrecognized Blend vs Mag Plots

**AddRomanLSST.ipynb**: Adds Roman images to LSST images by both upsampling and padding LSST Images giving us images of dims (512,512).

**galsim_truth_anns.ipynb (May not be as updated as `lsst_anns.py`)**: Notebook version of `lsst_anns.py` that creates annotations in _multiband_ for obsjs in cutouts using LSST Truth Catalog

**galsim_truth_anns_multiband.ipynb (Has been incorporated into `lsst_anns.py`)**: Grant's notebook adding multiband info to annotations

**RunInference.ipynb**: Notebook that evaluates the model and creates all the plots (AP scores, metrics, mags vs metrics, confusion matrices, etc).

**unrec_blend_frac.ipynb**: Notebook to calculate unrecognized blend fraction for both combined data trained model and non-combined data trained model.

**AllObjsHist.ipynb**: Plots histograms of distributions of Roman data.

### Folders

**data_processing/**: Contains all the data processing scripts/notebooks used to explore/format Roman data. The subfolder `lsst/` has the notebooks used to combine Roman + LSST data (starting with Roman data first and adding LSST images) and to create the LSST Detection catalog.