# Joint analysis of Roman/Rubin data with DeepDISC

## File Structure

**For HAL, OPEN THIS FILE WITH MARKDOWN READER in Jupyter Lab or Preview with Markdown Reader in HAL**

This section will explain hte structure and purpose of the folders/files in this repo.

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