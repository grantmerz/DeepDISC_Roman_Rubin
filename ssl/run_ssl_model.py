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
import json
# allows us to import from the custom configs directory w/o affecting deepdisc library imports
# update this to your config folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs')))
from custom.trainers import return_timed_lazy_trainer
from custom.hooks import return_timed_evallossHook
#from custom.meta_arch import return_frozen_model, return_unfrozen_model

import detectron2.utils.comm as comm

# import some common libraries
import numpy as np
import torch
import torch.distributed as dist

# import some common detectron2 utilities
from detectron2.config import LazyConfig
from detectron2.engine import launch

# Note: hsc_test_augs and train_augs don't exist in deepdisc
from deepdisc.data_format.image_readers import wlDC2ImageReader, wlHSCImageReader
from deepdisc.data_format.register_data import register_data_set
from deepdisc.model.loaders import return_test_loader, return_train_loader
from deepdisc.model.models import return_lazy_model
from custom.meta_arch_test import return_frozen_teacher_model

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


def main(args, freeze_teacher):
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
    #steps_per_epoch = cfg.dataloader.steps_per_epoch
    #e1 = steps_per_epoch * 15
    #e2 = steps_per_epoch * 25
    #e3 = steps_per_epoch * 35
    #efinal = steps_per_epoch * 50
    
    # for testing 
    steps_per_epoch = 10
    e1 = steps_per_epoch
    e2 = steps_per_epoch * 2
    e3 = steps_per_epoch * 3
    efinal = steps_per_epoch*4
    
    #Load the model
    model = return_lazy_model(cfg, freeze_teacher)
    
    # create data loaders
    kp=False
    mapper = cfg.dataloader.train.mapper(
        kp,cfg.dataloader.train.imagereader, cfg.dataloader.key_mapper, cfg.dataloader.augs
    ).map_data
    loader = return_train_loader(cfg, mapper)
    
    # for validation 
    # val_per = 10 for testing
    val_per = steps_per_epoch
    #khi = cfg.dataloader.test.keypoint_hflip_indices
    eval_mapper = cfg.dataloader.test.mapper(
        kp,cfg.dataloader.test.imagereader, cfg.dataloader.key_mapper, cfg.dataloader.augs
    ).map_data
    eval_loader = return_test_loader(cfg, eval_mapper)

    # set up the optimizer
    cfg.optimizer.params.model = model
    cfg.SOLVER.STEPS = [e1, e2, e3]
    cfg.SOLVER.MAX_ITER = efinal
    cfg.optimizer.lr = 0.001
    optimizer = return_optimizer(cfg)
    
    # choosing hooks for trainer
    saveHook = return_savehook(run_name, steps_per_epoch)
    schedulerHook = return_schedulerhook(optimizer)
    # lossHook = return_evallosshook(5, model, eval_loader)
    #lossHook = return_timed_evallossHook(val_per, model, eval_loader)
    
    #hookList = [lossHook, schedulerHook, saveHook]
    #no eval data set for testing
    hookList = [schedulerHook, saveHook]

    trainer = return_timed_lazy_trainer(model, loader, optimizer, cfg, hookList)
    trainer.set_period(steps_per_epoch // 2)
    trainer.train(0, cfg.SOLVER.MAX_ITER)
    # for testing
    # trainer.set_period(10)
    # trainer.train(0, 20) 
    if comm.is_main_process():
        #np.save(f"{output_dir}/{run_name}_losses", trainer.lossList)
        #np.save(f"{output_dir}/{run_name}_val_losses", trainer.vallossList)
        with open(os.path.join(output_dir,run_name) + "_losses.json", 'w') as json_file:
            json.dump(trainer.lossdict_epochs, json_file)
        with open(os.path.join(output_dir,run_name) + "_val_losses.json", 'w') as json_file:
            json.dump(trainer.vallossdict_epochs, json_file)
    
    # make sure all processes are done before destroying the process group
    comm.synchronize()
    # destroy the default process group (on every rank)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return

if __name__ == "__main__":

    '''
    This script will train a DeepDISC model with self-supervised learning.  

    A few things to note:
    
        1) We assume only a single modality, i.e., images from one telescope
        2) We use SupCon supervised contrastive learning to define positive/negative samples. 
            To do this, we need a unique objectID for each annotated object
            Positives are proposals with the same matched truth ID
            Negatives are all other proposals
        

    You need to use a demo_ssl_swin.py config file


    python run_model_job_30k_resume.py \
        --cfgfile ${CFG_FILE} \
        --train-metadata ${TRAIN_FILE} \
        --eval-metadata ${EVAL_FILE} \
        --num-gpus 4 \
        --run-name ${RUN_NAME} \
        --output-dir ${OUTPUT_DIR}

    '''


    args = make_training_arg_parser().parse_args()
    print("Command Line Args:", args)

    print("Training all layers with a SWIN Transformer backbone")
    # print("Running on GPUs:")
    torch.cuda.empty_cache()
    freeze_teacher = True
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(
            args,
            freeze_teacher
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(f"Took {time.time()-t0} seconds")

