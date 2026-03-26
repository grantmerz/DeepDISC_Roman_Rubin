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

import json
import gc
import os
import time
import sys
# allows us to import from the custom configs directory w/o affecting deepdisc library imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'deepdisc/configs')))
from custom.trainers import return_timed_lazy_trainer, EarlyStoppingException
from custom.hooks import return_timed_evallossHook, return_early_stoppingHook

import detectron2.utils.comm as comm

# import some common libraries
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system') # necessary for 100k training so we don't get "too many files open" error with dataloader workers > 0

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
    
    # Iterations for 5, 10, 15, 20 epochs
    steps_per_epoch = cfg.dataloader.steps_per_epoch
    e1 = steps_per_epoch * 5
    e2 = steps_per_epoch * 10
    e3 = steps_per_epoch * 15
    efinal = steps_per_epoch * 20

    val_per = steps_per_epoch
    model = return_lazy_model(cfg, freeze)
    mapper = cfg.dataloader.train.mapper(
        cfg.dataloader.train.imagereader,
        cfg.dataloader.key_mapper,
        cfg.dataloader.augs,
        keypoint_hflip_indices=[0],
    ).map_data
    loader = return_train_loader(cfg, mapper)
    
    eval_mapper = cfg.dataloader.test.mapper(
        cfg.dataloader.test.imagereader,
        cfg.dataloader.key_mapper,
        cfg.dataloader.augs,
        keypoint_hflip_indices=[0]
    ).map_data
    eval_loader = return_test_loader(cfg, eval_mapper)
    cfg.optimizer.params.model = model
    # if freeze:
    # setting up epochs and learning rate for training all layers
    cfg.SOLVER.STEPS = [e1,e2,e3]
    # cfg.SOLVER.STEPS = [e1]
    cfg.SOLVER.MAX_ITER = efinal
    
    cfg.optimizer.lr = 0.001
    optimizer = return_optimizer(cfg)
    
    # choosing hooks for trainer
    saveHook = return_savehook(run_name, steps_per_epoch)
    schedulerHook = return_schedulerhook(optimizer)
    # lossHook = return_evallosshook(5, model, eval_loader)
    lossHook = return_timed_evallossHook(val_per, model, eval_loader)
    # earlyStopHook = return_early_stoppingHook(
    #     patience=steps_per_epoch * 2,  # an epoch w/o improvement
    #     val_period=val_per,  # val period (every epoch)
    #     min_iters=steps_per_epoch * 5,  # don't activate early stopping until after 5 epochs
    #     save_best=True,  # save best model
    #     output_name=f"{run_name}_best"
    # )
    # earlyStopHook = return_early_stoppingHook(
    #     patience=2,  # an epoch w/o improvement
    #     val_period=2,  # val period (every epoch)
    #     min_delta=0.1,  # Loss must decrease by at least 0.1 to reset patience
    #     min_iters=5,  # don't activate early stopping until after 5 epochs
    #     save_best=True,  # save best model
    #     output_name=f"{run_name}_best_TEST"
    # )
    # hookList = [schedulerHook, saveHook]
    hookList = [lossHook, schedulerHook, saveHook]
    # hookList = [lossHook, earlyStopHook, schedulerHook, saveHook]
    
    trainer = return_timed_lazy_trainer(model, loader, optimizer, cfg, hookList)
    trainer.set_period(steps_per_epoch // 2)
    trainer.train(0, cfg.SOLVER.MAX_ITER)
    # trainer.set_period(5)
    # now we need to wrap training in a try-except to catch early stopping
    # early_stop = False
    # try:
    #     trainer.train(0, cfg.SOLVER.MAX_ITER)
    #     # trainer.train(0, 10) # for testing
    # except EarlyStoppingException as e:
    #     early_stop = True
    #     logger.info("****** EARLY STOPPING CAUGHT IN TRAINING SCRIPT ******")
    #     logger.info(f"Exception message: {e}")
    #     # synchonize processes after early stopping and before saving model
    #     # prevents deadlocks where other processes are still waiting while main process enters the except block
    #     comm.synchronize()             
    #     if comm.is_main_process():
    #         # save final model at iter where training stopped
    #         final_iter = trainer.iter + 1
    #         logger.info(f"Saving final model at iteration {final_iter}...")
    #         trainer.checkpointer.save(f"{run_name}_iter{final_iter}")
    #         logger.info(f"Final model saved as: {run_name}_iter{final_iter}.pth")
    #     comm.synchronize()
    if comm.is_main_process():
        # np.save(f"{output_dir}/{run_name}_losses", trainer.lossList)
        # np.save(f"{output_dir}/{run_name}_val_losses", trainer.vallossList)
        with open(os.path.join(output_dir,run_name) + "_losses.json", 'w') as json_file:
            json.dump(trainer.lossdict_epochs, json_file)
        with open(os.path.join(output_dir,run_name) + "_val_losses.json", 'w') as json_file:
            json.dump(trainer.vallossdict_epochs, json_file)
    # if comm.is_main_process():
    #     if early_stop:
    #         logger.info(f"Training Status: STOPPED EARLY")
    #         logger.info(f"Final iteration reached: {trainer.iter + 1} / {cfg.SOLVER.MAX_ITER}")
    #     else:
    #         logger.info(f"Training Status: COMPLETED FULLY")
    #         logger.info(f"Total iterations: {cfg.SOLVER.MAX_ITER}")
    #     np.save(f"{output_dir}/{run_name}_losses", trainer.lossList)
    #     np.save(f"{output_dir}/{run_name}_val_losses", trainer.vallossList)
    return
    # else:
        
    #     pass 
        # train the backbone as well after training the head layers and the stem layer
        # lower learning rate 0.0001 as we do not decay learning rate for retraining

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

