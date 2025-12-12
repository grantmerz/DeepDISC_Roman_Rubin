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

import gc
import os
import time
import sys
# allows us to import from the custom configs directory w/o affecting deepdisc library imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'deepdisc/configs')))
from custom.trainers import return_timed_lazy_trainer
from custom.hooks import return_timed_evallossHook

import detectron2.utils.comm as comm

# import some common libraries
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2.config import LazyConfig
from detectron2.engine import launch

# Note: hsc_test_augs and train_augs don't exist in deepdisc
from deepdisc.data_format.image_readers import wlDC2ImageReader, wlHSCImageReader
from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import return_test_loader, return_train_loader
from deepdisc.model.models import return_lazy_model
from deepdisc.training.trainers import (
    return_evallosshook,
    return_lazy_trainer,
    return_optimizer,
    return_savehook,
    return_schedulerhook
)
from deepdisc.utils.parse_arguments import make_training_arg_parser
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def main(args, freeze):
    # Hack if you get SSL certificate error
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    logger = setup_logger()
    # Handle args
    output_dir = args.output_dir
    run_name = args.run_name    

    # Get file locations
    trainfile = args.train_metadata
    evalfile = args.eval_metadata

    cfgfile = args.cfgfile
    
    # Load the config
    cfg = LazyConfig.load(cfgfile)
    for key in cfg.get("MISC", dict()).keys():
        cfg[key] = cfg.MISC[key]
    
    # if args.num_gpus==1 and not freeze:
    #     DatasetCatalog.remove(cfg.DATASETS.TRAIN)
    #     MetadataCatalog.remove(cfg.DATASETS.TRAIN)
    #     DatasetCatalog.remove(cfg.DATASETS.TEST)
    #     MetadataCatalog.remove(cfg.DATASETS.TEST)
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
    
    # Iterations for 15, 25, 35, 50 epochs
    # steps_per_epoch = cfg.dataloader.steps_per_epoch
    steps_per_epoch = 2
    e1 = steps_per_epoch * 15
    # e2 = steps_per_epoch * 25
    # e3 = steps_per_epoch * 35
    efinal = steps_per_epoch * 50
    # e1 = steps_per_epoch
    # e2 = steps_per_epoch * 2
    # efinal = e2
    # e3 = steps_per_epoch * 35
    # efinal = steps_per_epoch * 50
    model = return_lazy_model(cfg, freeze)
    # for training
    mapper = cfg.dataloader.train.mapper(
        cfg.dataloader.train.imagereader, cfg.dataloader.key_mapper, cfg.dataloader.augs
    ).map_data
    loader = return_train_loader(cfg, mapper)
    # for validation 
    # val_per = steps_per_epoch
    val_per = 10
    eval_mapper = cfg.dataloader.test.mapper(
        cfg.dataloader.test.imagereader, cfg.dataloader.key_mapper, cfg.dataloader.augs
    ).map_data
    eval_loader = return_test_loader(cfg, eval_mapper)
    cfg.optimizer.params.model = model
    # if freeze:
    # setting up epochs and learning rate for training all layers
    # cfg.SOLVER.STEPS = [e1,e2,e3]
    cfg.SOLVER.STEPS = [e1]
    cfg.SOLVER.MAX_ITER = efinal
    
    cfg.optimizer.lr = 0.001
    optimizer = return_optimizer(cfg)
    
    # choosing hooks for trainer
    saveHook = return_savehook(run_name, steps_per_epoch)
    schedulerHook = return_schedulerhook(optimizer)
    # lossHook = return_evallosshook(5, model, eval_loader)
    lossHook = return_timed_evallossHook(val_per, model, eval_loader)
    
    # hookList = [schedulerHook, saveHook]
    hookList = [lossHook, schedulerHook, saveHook]
    
    trainer = return_timed_lazy_trainer(model, loader, optimizer, cfg, hookList)
    # trainer.set_period(steps_per_epoch // 2)
    # trainer.train(0, cfg.SOLVER.MAX_ITER)
    trainer.set_period(5)
    trainer.train(0, 20) # for testing
    if comm.is_main_process():
        np.save(f"{output_dir}/{run_name}_losses", trainer.lossList)
        np.save(f"{output_dir}/{run_name}_val_losses", trainer.vallossList)
    
    # to avoid this warning:
    # [rank0]:[W1212 09:05:12.615562415 ProcessGroupNCCL.cpp:1524] 
    # Warning: WARNING: destroy_process_group() was not called before program exit, 
    # which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
    
    # make sure all processes are done before destroying the process group
    comm.synchronize()
    # destroy the default process group (on every rank)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return

if __name__ == "__main__":
    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)

    print("Training all layers with a SWIN Transformer backbone")
    # print("Running on GPUs:")
    torch.cuda.empty_cache()
    freeze = False
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

    torch.cuda.empty_cache()
    gc.collect()

    print(f"Took {time.time()-t0} seconds")
