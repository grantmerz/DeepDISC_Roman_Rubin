try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=sShapelyDeprecationWarning)
except:
    pass
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

import gc
import os
import time

import detectron2.utils.comm as comm
import detectron2.data as d2data

# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2.config import LazyConfig, get_cfg
from detectron2.engine import launch

from deepdisc.data_format.augment_image import hsc_test_augs, train_augs
from deepdisc.data_format.image_readers import DC2ImageReader, HSCImageReader, RomanImageReader
from deepdisc.data_format.register_data import register_data_set
import deepdisc.model.loaders as loaders 
from deepdisc.model.loaders import return_test_loader, return_train_loader
from deepdisc.model.models import RedshiftPDFCasROIHeads, return_lazy_model
from deepdisc.training.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook,
)
from deepdisc.utils.parse_arguments import dtype_from_args, make_training_arg_parser


def main(args):
    # Hack if you get SSL certificate error
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Handle args
    output_dir = args.output_dir
    output_name = args.run_name    

    # Get file locations
    trainfile = args.train_metadata
    evalfile = args.eval_metadata

    cfgfile = args.cfgfile
    
    # Load the config
    cfg = LazyConfig.load(cfgfile)
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]

    # Register the data sets
    astrotrain_metadata = register_data_set(
        cfg.DATASETS.TRAIN, trainfile, thing_classes=cfg.metadata.classes
    )
    astroval_metadata = register_data_set(
        cfg.DATASETS.TEST, evalfile, thing_classes=cfg.metadata.classes
    )
    
    # Set the output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    epoch = cfg.dataloader.epoch

    val_per = epoch

    model = return_lazy_model(cfg)

    cfg.optimizer.params.model = model
    cfg.optimizer.lr = 0.001
    optimizer = return_optimizer(cfg)
    
    #change this depending on the metadata dict
    def roman_key_mapper(dataset_dict): # needs to return where the cutout image data for each cutout is stored 
        fn = dataset_dict["file_name"]
        return fn


    mapper = cfg.dataloader.train.mapper(
        cfg.dataloader.imagereader, roman_key_mapper, cfg.dataloader.augs
    ).map_data


    loader = return_train_loader(cfg, mapper)
    eval_loader = return_test_loader(cfg, mapper)

    saveHook = return_savehook(output_name)
    lossHook = return_evallosshook(val_per, model, eval_loader)
    schedulerHook = return_schedulerhook(optimizer)
    hookList = [lossHook, schedulerHook, saveHook]

    trainer = return_lazy_trainer(model, loader, optimizer, cfg, hookList)
#     trainer.set_period(5)
#     trainer.train(0, 10)
    trainer.set_period(epoch//2)
    trainer.train(0, cfg.SOLVER.MAX_ITER)
    if comm.is_main_process():
        np.save(f"{output_dir}/{output_name}_losses", trainer.lossList)
        np.save(f"{output_dir}/{output_name}_val_losses", trainer.vallossList)
    return
    

if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)

    print("Training head layers")
    print("Running on GPUs:")
    torch.cuda.empty_cache()
    train_head = True
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            args,
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()

    
    print(f"Took {time.time()-t0} seconds")

# run for a single patch
#     python run_model.py --cfgfile ./deepdisc/configs/solo/solo_swin_Roman.py --train-metadata roman_data/annotations/train_dc2_50.93_-42.0.json --eval-metadata roman_data/annotations/eval_dc2_50.93_-42.0.json --num-gpus 2 --run-name first_official_test --output-dir ./roman_runs

# run just for a few iterations
#     python run_model.py --cfgfile ./deepdisc/configs/solo/solo_swin_Roman.py --train-metadata roman_data/annotations/train_dc2_50.93_-42.0.json --eval-metadata roman_data/annotations/eval_dc2_50.93_-42.0.json --num-gpus 2 --run-name random --output-dir .

# run for all patches
#     python run_model.py --cfgfile ./deepdisc/configs/solo/solo_swin_Roman.py --train-metadata roman_data/annotations/train_roman.json --eval-metadata roman_data/annotations/val_roman.json --num-gpus 2 --run-name all_roman --output-dir ./roman_runs/run10-full

# for lsst data 3828
# python run_model.py --cfgfile ./deepdisc/configs/solo/swin_lsst.py --train-metadata lsst_data/annotations/train.json --eval-metadata lsst_data/annotations/val.json --num-gpus 2 --run-name lsst_256 --output-dir ./lsst_runs/run1_sm_dc2

# for lsst data overlap
# python run_model.py --cfgfile ./deepdisc/configs/solo/swin_lsst.py --train-metadata lsst_data/annotations/train.json --eval-metadata lsst_data/annotations/val.json --num-gpus 2 --run-name lsst --output-dir ./lsst_runs/run2_sm