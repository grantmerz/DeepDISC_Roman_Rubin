import time
import numpy as np
import torch
import torch.distributed as dist

from detectron2.utils.logger import setup_logger
from detectron2.engine import HookBase
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from deepdisc.astrodet.detectron import LossEvalHook

class TimedLossEvalHook(LossEvalHook):

    """
    Validation loss hook with timing info
    Parameters
    ----------
    eval_period: int
        How many iterations to run before validation loss is calculated
    model: torch.NN.module
        The model being trained
    data_loader: detectron2 DataLoader
        The dataloader that loads in the evaluation dataset
    """

    def __init__(self, eval_period, model, data_loader):
        super().__init__(eval_period, model, data_loader)
        self.times = {
            'inference_times': [], # per batch inference times
            'total_times': [], # total time per validation run
            'num_batches': [], # num batches per validation run
        }
        self.logger = setup_logger()

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        self.logger.info(f"Starting validation on: {total} batches")
        num_warmup = min(5, total - 1)
        
        start_time = time.perf_counter()
        total_compute_time = 0
        
        batch_inference_times = []
        losses = []
        losses_dicts = []
        
        # was_training = self._model.training # save current mode
        # self._model.train() # to ensure we can do loss computation

        with torch.no_grad():
            for idx, inputs in enumerate(self._data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0
                    batch_inference_times = []
                
                infer_start = time.perf_counter()
                # ensure that all CUDA ops are done before starting timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                loss_batch, metrics_dict = self._get_loss(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_time = time.perf_counter() - infer_start
                
                if idx >= num_warmup:
                    batch_inference_times.append(infer_time)
                    total_compute_time += infer_time
                
                losses.append(loss_batch)
                losses_dicts.append(metrics_dict)

                # logs every 25% regardless of warmup
                log_interval = max(total // 4, 1)
                if (idx + 1) % log_interval == 0 or (idx + 1) == total:
                    elapsed = time.perf_counter() - start_time
                    avg = elapsed / (idx + 1)
                    eta = avg * (total - idx - 1)
                    
                    warmup_marker = " [WARMUP]" if idx < num_warmup else ""
                    self.logger.info(
                        f"Progress: {idx + 1}/{total} ({(idx+1)/total*100:.0f}%){warmup_marker} |"
                        f" Avg: {avg:.3f}s/batch | ETA: {eta:.1f}s"
                    )
        # if not was_training:
        #     self._model.eval()
        total_time = time.perf_counter() - start_time
        
        # stats using post-warmup batches only
        avg_infer_time = np.mean(batch_inference_times)
        total_infer_time = np.sum(batch_inference_times)
        data_time = total_time - total_infer_time
        # computing the losses
        mean_loss = np.mean(losses)
        averaged_losses_dict = {}
        for d in losses_dicts:
            for key, value in d.items():
                if key not in averaged_losses_dict:
                    averaged_losses_dict[key] = [0, 0]  # [sum, count]
                averaged_losses_dict[key][0] += value
                averaged_losses_dict[key][1] += 1
        averaged_losses_dict = {key: total / count for key, (total, count) in averaged_losses_dict.items()}
        # print('validation_loss', mean_loss)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        self.trainer.add_val_loss(mean_loss)
        self.trainer.valloss = mean_loss
        self.trainer.add_val_loss_dict(averaged_losses_dict)
        self.trainer.vallossdict = averaged_losses_dict
        
        # storing times
        self.times['inference_times'].append(batch_inference_times)
        self.times['total_times'].append(total_time)
        self.times['num_batches'].append(len(batch_inference_times))
        
        # logging all times
        self.logger.info("Validation done!")
        self.logger.info(f" Total time: {total_time:.3f}s")
        self.logger.info(f" Data loading: {data_time:.3f}s ({data_time/total_time*100:.1f}%)")
        self.logger.info(f" Inference (post-warmup): {total_infer_time:.3f}s ({total_infer_time/total_time*100:.1f}%)")
        self.logger.info(f" Batches (post-warmup): {len(batch_inference_times)}")
        self.logger.info(f" Avg per batch: {avg_infer_time}s")
        self.logger.info(f" Mean val loss: {mean_loss:.5f}")
        comm.synchronize()
        return losses

def return_timed_evallossHook(val_per, model, test_loader):
    """Returns a hook for evaulating the loss with timing
    Parameters
    ----------
    val_per : int
        the frequency with which to calculate validation loss
    model: torch.nn.module
        the model
    test_loader: data loader
        the loader to read in the eval data

    Returns
    -------
        a TimedLossEvalHook
    """
    timed_lossHook = TimedLossEvalHook(val_per, model, test_loader)
    return timed_lossHook

    # TODO: we shld adapt for other metrics user wants to track
class EarlyStoppingHook(HookBase):
    """
    Hook for early stopping during training based off of validation loss improvements. 
    If no improvement is seen for a specified number of iterations (patience), training stops.
    
    Parameters
    ----------
    patience : int
        Number of iters without improvement before stopping training
    val_period : int
        How often validation runs (in iters)
    min_delta : float, optional
        Val loss must decrease by at least min_delta to qualify as an improvement and reset patience (default: 0.001)
    min_iters : int, optional
        Minimum number of iters before early stopping can activate (default: 100)
    save_best : bool, optional
        Whether to save the best model checkpoint (default: True)
    output_name : str, optional
        Name prefix for saved checkpoints (default: "best")
    """
    
    def __init__(
        self, 
        patience, 
        val_period,
        min_delta=0.001,
        min_iters=100,
        save_best=True,
        output_name="best"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_iters = min_iters
        self.val_period = val_period
        self.save_best = save_best
        self.output_name = output_name
        
        self.best_loss = np.inf
        self.best_iter = 0
        self.iters_no_improvement = 0
        self.early_stop_flag = False
        self.logger = setup_logger()
        
    def before_train(self):
        if comm.is_main_process():
            self.logger.info("Early Stopping Config:")
            self.logger.info(f" Patience: {self.patience} iters")
            self.logger.info(f" Min delta: {self.min_delta}")
            self.logger.info(f" Min iters before activation: {self.min_iters}")
            self.logger.info(f" Save best model: {self.save_best}")
    # checking for early stopping after each training step
    def after_step(self):
        # only main process does logic, others just wait
        if comm.is_main_process():
            # only main process does the logic
            next_iter = self.trainer.iter + 1 # count 1-based so its more intuitive
            # checking if validation just ran (val happens when next_iter % period == 0)
            if self.val_period and next_iter % self.val_period == 0:
                # skip it if we haven't reached min iters
                if next_iter < self.min_iters:
                    if next_iter == self.val_period:  # log only on first validation
                        self.logger.info(
                            f"Early stopping inactive until iter {self.min_iters} "
                            f"(currently at {next_iter})"
                        )
                else:
                    current_val_loss = self.trainer.valloss
                    if current_val_loss < (self.best_loss - self.min_delta): # checking for improvement
                        # we have an improvement
                        improvement = self.best_loss - current_val_loss
                        self.best_loss = current_val_loss
                        self.best_iter = next_iter
                        self.iters_no_improvement = 0                
                        self.logger.info(f"-----NEW BEST MODEL at iter {next_iter}-----")
                        self.logger.info(f" Best val loss: {self.best_loss:.6f} (improved by {improvement:.6f})")
                        self.logger.info(f" Patience counter reset to 0")              
                        # saving best model to disk
                        if self.save_best:
                            self.logger.info(f" Saving best model as '{self.output_name}'...")
                            self.trainer.checkpointer.save(self.output_name)
                    else:
                        # we don't have an improvement
                        self.iters_no_improvement += self.val_period
                        self.logger.info(f"Early Stopping Status at iter {next_iter}:")
                        self.logger.info(f" Current val loss: {current_val_loss:.6f}")
                        self.logger.info(f" Best val loss: {self.best_loss:.6f} (at iter {self.best_iter})")
                        self.logger.info(f" Patience: {self.iters_no_improvement}/{self.patience} iters without improvement")
                        # did we exceed patience?
                        if self.iters_no_improvement >= self.patience:
                            self.early_stop_flag = True
                            self.trainer.early_stop = True
                            self.logger.info("***** EARLY STOPPING TRIGGERED!!!! *****")
                            self.logger.info(f" No improvement for {self.iters_no_improvement} iters")
                            self.logger.info(f" Best val loss: {self.best_loss:.6f} (at iter {self.best_iter})")
                            self.logger.info(f" Final val loss: {current_val_loss:.6f} (at iter {next_iter})")
                            self.logger.info(f" Training will stop after this iter")
        # now all processes run the below
        comm.synchronize() # synchronize all processes so rank0 has finished its check
        # check if we're in a distributed setting and if not, we're done as the main process's early_stop flag is what matters
        if not dist.is_available() or not dist.is_initialized() or comm.get_world_size() == 1:
            return
        # if we are distributed, now we need to broadcast early stopping flag to all processes so they can stop too
        # creates the tensor but only rank0's value is correct
        if torch.cuda.is_available():
            device = torch.device(comm.get_local_rank()) # use process's local rank to place tensor on correct GPU
            stop_signal = torch.tensor(int(self.early_stop_flag), device=device, dtype=torch.int)
        else:
            stop_signal = torch.tensor(int(self.early_stop_flag), dtype=torch.int)
        # broadcasting signal from rank 0 to all other processes
        dist.broadcast(stop_signal, src=0)
        # all processes check the signal
        if stop_signal.item() == 1:
            self.trainer.early_stop = True # setting local trainer's early stop flag
    
        comm.synchronize() # ensure all processes are synced before proceeding
        
    def after_train(self):
        if comm.is_main_process():
            self.logger.info("Early Stopping Summary:")
            if self.early_stop_flag:
                self.logger.info(" Status: Training stopped early")
            else:
                self.logger.info(" Status: Training completed without early stopping")
            self.logger.info(f"Best val loss: {self.best_loss:.6f} (at iter {self.best_iter})")

def return_early_stoppingHook(
    patience,
    val_period,
    min_delta=0.001,
    min_iters=100,
    save_best=True,
    output_name="best"
):
    """Returns an early stopping hook
    
    Parameters
    ----------
    patience : int
        Number of iters without improvement before stopping training
    val_period : int
        How often validation runs (in iters)
    min_delta : float, optional
        Val loss must decrease by at least min_delta to qualify as an improvement and reset patience (default: 0.001)
    min_iters : int, optional
        Minimum number of iters before early stopping can activate (default: 100)
    save_best : bool, optional
        Whether to save the best model checkpoint (default: True)
    output_name : str, optional
        Name prefix for saved checkpoints (default: "best")
    
    Returns
    -------
    EarlyStoppingHook
        A configured early stopping hook
    """
    return EarlyStoppingHook(
        patience=patience,
        val_period=val_period,
        min_delta=min_delta,
        min_iters=min_iters,
        save_best=save_best,
        output_name=output_name
    )
