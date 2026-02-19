import json
import time

import torch
import numpy as np
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from deepdisc.training.trainers import LazyAstroTrainer

class EarlyStoppingException(Exception):
    """Custom exception raised when early stopping is triggered"""
    pass

class TimedLazyAstroTrainer(LazyAstroTrainer):
    def __init__(self, model, data_loader, optimizer, cfg):
        super().__init__(model, data_loader, optimizer, cfg)
        
        self.times = {
            'data_times': [],
            'loss_times': [],
            'backward_times': [],
            'step_times': [],
            'total_times': []
        }
        
        self.sum_times = {
            'data_time': 0.0,
            'loss_time': 0.0,
            'backward_time': 0.0,
            'step_time': 0.0,
            'total_time': 0.0
        }

        self.timing_report_period = cfg.train.timing_report_period
        self.timing_rolling_window_size = cfg.train.timing_rolling_window_size
        self.timing_save_period = cfg.train.timing_save_period
        self.output_dir = cfg.OUTPUT_DIR
        self.logger = setup_logger()
        
        self.start_time = None
        self.early_stop = False
    
    # start timing right before training begins
    def before_train(self):
        super().before_train()
        self.start_time = time.perf_counter()
        if comm.is_main_process():
            self.logger.info("Using TimedLazyAstroTrainer to display and record timing info...")
            self.logger.info(f" Report Period: {self.timing_report_period} iters")
            self.logger.info(f" Rolling Average over last: {self.timing_rolling_window_size} iters")
            self.logger.info(f" Save Period: {self.timing_save_period} iters")

    # Adding timing info to each step
    def run_step(self):
        # check if early stopping was triggered
        if self.early_stop:
            raise EarlyStoppingException("Training stopped early - Patience exceeded.")
        
        self.iterCount = self.iterCount + 1
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        
        # Data loading time
        data_start = time.perf_counter()
        data = next(self._data_loader_iter)
        self.latest_data = data
        data_time = time.perf_counter() - data_start
        # Forward pass time
        loss_start = time.perf_counter()
        loss_dict = self.model(data)
        loss_time = time.perf_counter() - loss_start
        # same as LazyAstroTrainer run_step 
        ld = {
             k: v.detach().cpu().item() if (isinstance(v, torch.Tensor) and v.numel()==1)  else v.tolist()
             for k, v in loss_dict.items()
        }
        self.lossdict_epochs[str(self.iter+1)] = ld
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
            all_losses = [l.cpu().detach().item() for l in loss_dict.values()]
        # Backward pass time
        backward_start = time.perf_counter()
        self.optimizer.zero_grad()
        losses.backward()
        backward_time = time.perf_counter() - backward_start
        # optimizer step time
        step_start = time.perf_counter()
        self.optimizer.step()
        step_time = time.perf_counter() - step_start
        
        total_time = time.perf_counter() - start
        
        self.times['data_times'].append(data_time)
        self.times['loss_times'].append(loss_time)
        self.times['backward_times'].append(backward_time)
        self.times['step_times'].append(step_time)
        self.times['total_times'].append(total_time)
        
        self.sum_times['data_time'] += data_time
        self.sum_times['loss_time'] += loss_time
        self.sum_times['backward_time'] += backward_time
        self.sum_times['step_time'] += step_time
        self.sum_times['total_time'] += total_time
        # just storing the losses as LazyAstroTrainer does
        self.lossList.append(losses.cpu().detach().numpy())
        
        # every p iters that user specifies in training script, we print timing info for that iter
        if self.iter % self.period == 0 and comm.is_main_process():
            print(
                f"Iteration: {self.iter} | "
                f"data: {data_time:.5f}s | "
                f"forward: {loss_time:.5f}s | "
                f"backward: {backward_time:.5f}s | "
                f"optimizer step: {step_time:.5f}s | "
                f"Total: {total_time:.5f}s | "
                f"loss keys: {loss_dict.keys()} | "
                f"losses: {all_losses} | "
                f"Val Loss: {self.valloss} | "
                f"lr: {self.scheduler.get_lr()}"
            )
        # now we report rolling window avgs over last window_size iters every report period
        if self.iter % self.timing_report_period == 0 and comm.is_main_process():
            self._print_timing()
        # and save timing info every save period
        if self.iter % self.timing_save_period == 0 and comm.is_main_process():
            self._save_timing()
    
    # internal function to print rolling window timing stats
    def _print_timing(self):
        start = max(0, len(self.times['total_times']) - self.timing_rolling_window_size)
        if len(self.times['total_times'][start:]) == 0:
            self.logger.warning("No iters completed, cannot print timing stats")
            return
        avg_data = np.mean(self.times['data_times'][start:])
        avg_loss = np.mean(self.times['loss_times'][start:])
        avg_backward = np.mean(self.times['backward_times'][start:])
        avg_step = np.mean(self.times['step_times'][start:])
        avg_total = np.mean(self.times['total_times'][start:])
        
        self.logger.info(f"Timing Report at Iter {self.iter} - Avg over last {len(self.times['total_times'][start:])} iters:")
        self.logger.info(f" Data Loading: {avg_data:.5f}s ({avg_data/avg_total*100:.2f}%)")
        self.logger.info(f" Forward Pass: {avg_loss:.5f}s ({avg_loss/avg_total*100:.2f}%)")
        self.logger.info(f" Backward Pass: {avg_backward:.5f}s ({avg_backward/avg_total*100:.2f}%)")
        self.logger.info(f" Optimizer Step: {avg_step:.5f}s ({avg_step/avg_total*100:.2f}%)")
        self.logger.info(f" Total Iter: {avg_total:.5f}s")

    # internal function to save timing info
    def _save_timing(self, last=False):
        suffix = "_last" if last else f"_iter{self.iter}"
        summary = {
            'iter': self.iter,
            'total_train_time': time.perf_counter() - self.start_time,
            'num_iters': len(self.times['total_times']),
            'sum': {
                'data': self.sum_times['data_time'],
                'forward': self.sum_times['loss_time'],
                'backward': self.sum_times['backward_time'],
                'optimizer_step': self.sum_times['step_time'],
                'total': self.sum_times['total_time']
            },
            'avgs': { # change to python float so JSON serializable
                'data': float(np.mean(self.times['data_times'])) if self.times['data_times'] else 0,
                'forward': float(np.mean(self.times['loss_times'])) if self.times['loss_times'] else 0,
                'backward': float(np.mean(self.times['backward_times'])) if self.times['backward_times'] else 0,
                'optimizer_step': float(np.mean(self.times['step_times'])) if self.times['step_times'] else 0,
                'total': float(np.mean(self.times['total_times'])) if self.times['total_times'] else 0
            }
        }
        summary_file = f"{self.output_dir}/timing{suffix}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Saved timing summary to {summary_file}")
        # only save time arrays on the final save after training is complete
        if last:
            np.save(f"{self.output_dir}/data_times.npy", np.array(self.times['data_times']))
            np.save(f"{self.output_dir}/forward_times.npy", np.array(self.times['loss_times']))
            np.save(f"{self.output_dir}/backward_times.npy", np.array(self.times['backward_times']))
            np.save(f"{self.output_dir}/optimizer_step_times.npy", np.array(self.times['step_times']))
            np.save(f"{self.output_dir}/total_times.npy", np.array(self.times['total_times']))
            self.logger.info(f"Saved time arrays to {self.output_dir}")

    # print timing info after training is complete and save the arrays
    def after_train(self):
        super().after_train()
        if comm.is_main_process():
            self._print_final_timing()
            self._save_timing(last=True)
    
    # internal function to print final timing report
    def _print_final_timing(self):
        total_time = time.perf_counter() - self.start_time
        num_iters = len(self.times['total_times'])
        if num_iters == 0:
            self.logger.warning("No iters completed, cannot print timing stats")
            return
        self.logger.info(f"------- FINAL TIMING REPORT -------")
        self.logger.info(f"Total Training Time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        self.logger.info(f"Total Iters: {num_iters}")
        self.logger.info(f"Cumulative Times:")
        self.logger.info(f" Data Loading: {self.sum_times['data_time']:.2f}s "
                         f"({self.sum_times['data_time']/total_time*100:.1f}%)")
        self.logger.info(f" Forward Pass: {self.sum_times['loss_time']:.2f}s "
                         f"({self.sum_times['loss_time']/total_time*100:.1f}%)")
        self.logger.info(f" Backward Pass: {self.sum_times['backward_time']:.2f}s "
                         f"({self.sum_times['backward_time']/total_time*100:.1f}%)")
        self.logger.info(f" Optimizer Step: {self.sum_times['step_time']:.2f}s "
                         f"({self.sum_times['step_time']/total_time*100:.1f}%)")
        self.logger.info(f" Iters: {self.sum_times['total_time']:.2f}s "
                         f"({self.sum_times['total_time']/total_time*100:.1f}%)")
        
        self.logger.info(f"Avg Per Iter:")
        self.logger.info(f" Data Loading:   {np.mean(self.times['data_times']):.4f}s")
        self.logger.info(f" Forward Pass:   {np.mean(self.times['loss_times']):.4f}s")
        self.logger.info(f" Backward Pass:  {np.mean(self.times['backward_times']):.4f}s")
        self.logger.info(f" Optimizer Step: {np.mean(self.times['step_times']):.4f}s")
        self.logger.info(f" Total:          {np.mean(self.times['total_times']):.4f}s")
        self.logger.info(f"Throughput:")
        self.logger.info(f" Iters/sec: {num_iters/total_time:.2f}")
        self.logger.info(f" Sec/iter:  {total_time/num_iters:.2f}")

def return_timed_lazy_trainer(model, loader, optimizer, cfg, hooklist):
    """Return a trainer that has timing info for models built on LazyConfigs
    Parameters
    ----------
    model : torch model
        pointer to file
    loader : detectron2 data loader

    optimizer : detectron2 optimizer

    cfg : .py file
        The LazyConfig used to build the model, and also stores config vals for data loaders

    hooklist : list
        The list of hooks to use for the trainer

    Returns
    -------
        trainer
    """
    trainer = TimedLazyAstroTrainer(model, loader, optimizer, cfg)
    if hooklist:
        trainer.register_hooks(hooklist)
    else:
        print("No hooks to register!")
    return trainer