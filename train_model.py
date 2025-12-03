try:
    # ignore ShapelyDeprecationWarning from fvcore                                                                
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
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
from detectron2.config import LazyConfig, get_cfg, instantiate
from detectron2.engine import launch
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
import pickle as pkl
import detectron2.data as data
from configs.custom.meta_arch import GeneralizedRCNNMoco
from deepdisc.utils.parse_arguments import dtype_from_args, make_training_arg_parser
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook,
)


def main(args,freeze):
    # Hack if you get SSL certificate error
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Handle args
    output_dir = '.'
    #run_name = args.run_name  
    #bs = args.batch_size
    bs = 2
    # Get file locations
    #trainfile = args.train_metadata
    #evalfile = args.eval_metadata
   

    cfgfile= '/work/hdd/bdsp/g4merz/DeepDISC_Roman_Rubin/DeepDISC_Roman_Rubin/configs/solo/swin_lsst_moco.py'
    
    # Load the config
    cfg = LazyConfig.load(cfgfile)
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]
    
    # Set the output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    #Set batch size
    cfg.SOLVER.IMS_PER_BATCH = bs
    cfg.dataloader.train.total_batch_size = bs


    #cfg.train.init_checkpoint='/projects/bdsp/yse2/cascade_mask_rcnn_swin_b_in21k_model.pkl'
    cfg.train.init_checkpoint=None


    # Iterations for 15, 25, 35, 50 epochs
    epoch = int(6778/bs)
    e1 = epoch * 15
    e2 = epoch * 25
    e3 = epoch * 35
    efinal = epoch * 50

    #epoch= 4
    #e1 = epoch * 5
    #e2 = epoch * 10
    #e3 = epoch * 15
    #efinal = epoch * 20
    
    #epoch=175
    #e1 = epoch * 30
    #e2 = epoch * 30
    #e3 = epoch * 60
    #efinal = epoch * 120

    cfg.train.ddp.find_unused_parameters=False

    #locate where the gradient vanishes 

    model = return_lazy_model(cfg,freeze=False)

    #model = instantiate(cfg.model)

    cfg.optimizer.params.model = model


    filepath = '/projects/bdsp/truth-lsst/grant_4k.json'
    from deepdisc.data_format.file_io import DDLoader
    json_loader = DDLoader()
    ddicts = json_loader.load_coco_json_file(filepath).get_dataset()

    ddicts_test =[]
    for i,d in enumerate(ddicts):
        if "51.52_-39.1" in d['file_name']:
            print(i)
            ddicts_test.append(d)


    def lsst_key_mapper(dataset_dict):
        key = dataset_dict["file_name"]
        im = os.path.basename(key)
        k = os.path.join("/projects/bdsp/truth-lsst/51.52_-39.1/",im)
        return k
    
    def rubin_roman_key_mapper(dataset_dict):
        key = dataset_dict["file_name"]
        im = os.path.basename(key)
        lsst_key = os.path.join("/projects/bdsp/truth-lsst/51.52_-39.1/",im)
        roman_key = os.path.join("/projects/bdsp/truth-roman/51.52_-39.1/",im)

        return lsst_key,roman_key


    from configs.custom.image_readers import RomanRubinImageReader
    
    IR = RomanRubinImageReader()

    mapper = cfg.dataloader.train.mapper(
            IR, rubin_roman_key_mapper, cfg.dataloader.augs
        ).map_data
    
    optimizer = return_optimizer(cfg)

    loader = data.build_detection_train_loader(ddicts_test,mapper=mapper,total_batch_size=bs)

    #batch = next(iter(loader))

    hooklist = []

    trainer = return_lazy_trainer(model, loader, optimizer, cfg, hooklist)
    trainer.set_period(5)
    trainer.train(0, 20)
    
    return
            
    

if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)
    freeze=False
    print("Forward pass")
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            args,
            freeze
        ),
    )


