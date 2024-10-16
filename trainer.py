import os
import time
import json
import logging
import datetime
from typing import List
import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from models.base_model import BaseModel
from dataset.dataloader_utils import MultiIterLoader, IterLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import ChainDataset

from common.utils import (
    get_rank,
    is_main_process,
    is_dist_avail_and_initialized,
    main_process,
    prepare_sample,
    get_world_size,
    get_worker_init_fn,
    append_metrics_to_csv,
)
from common.logger import MetricLogger, SmoothedValue
from common.config import LLMConfig
from common.optims_origin import LinearWarmupCosineLRScheduler
from evaluators.affordance_eval import (
    AffordanceAccEval,
)  # Used for import evaluators,then generate _pycache_


class Trainer:
    def __init__(
        self,
        cfg: LLMConfig,
        job_id: str,
        accelerator: Accelerator,
        model: BaseModel,
        train_datasets: List[Dataset],
        eval_datasets: List[Dataset],
        task_evals: List,
    ) -> None:
        self.cfg = cfg
        self.job_id = job_id
        self.accelerator = accelerator
        self._model = model
        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets
        self.device = torch.device(self.cfg.device)
        self.task_evals = task_evals

        self._build_optimizer()
        self._build_train_loader()
        self._build_eval_loader()
        self._prepare_all()

        self.start_epoch = 0

    def _build_optimizer(self):
        lr_scale = self.cfg.run_cfg.get("lr_layer_decay", 1)
        weight_decay = self.cfg.run_cfg.get("weight_decay", 0.05)
        beta2 = self.cfg.run_cfg.get("beta2", 0.999)
        self.optimizer = torch.optim.AdamW(
            self._model.get_optimizer_params(weight_decay, lr_scale),
            lr=float(self.cfg.run_cfg.init_lr),
            betas=(0.9, beta2),
        )

        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            optimizer=self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.cfg.run_cfg.iters_per_epoch,
            min_lr=float(self.cfg.run_cfg.min_lr),
            init_lr=float(self.cfg.run_cfg.init_lr),
            decay_rate=self.cfg.run_cfg.get("lr_decay_rate", None),
            warmup_start_lr=self.cfg.run_cfg.get("warmup_lr", -1),
            warmup_steps=self.cfg.run_cfg.get("warmup_steps", 0),
        )

    def _build_eval_loader(self):
        data_loaders = []
        for dataset in self.eval_datasets:
            collate_fn = dataset.collate if hasattr(dataset, "collate") else None
            loader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.cfg.run_cfg.batch_size_eval,
                pin_memory=True,
                num_workers=self.cfg.run_cfg.num_workers,
                collate_fn=collate_fn,
                worker_init_fn=get_worker_init_fn(self.cfg.run_cfg.seed),
            )
            data_loaders.append(loader)
        self.eval_loaders = data_loaders

    def _build_train_loader(self):
        if len(self.train_datasets) > 0:
            dataloaders = []
            ratios = []
            for ds in self.train_datasets:
                collate_fn = ds.collate if hasattr(ds, "collate") else None

                if isinstance(ds, ChainDataset) or isinstance(ds, wds.DataPipeline):
                    # wds.WebdDataset instance are chained together
                    # webdataset.DataPipeline has its own sampler and collate_fn
                    loader = iter(
                        DataLoader(
                            ds,
                            batch_size=self.cfg.run_cfg.batch_size_train,
                            num_workers=self.cfg.run_cfg.num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            # seed
                            worker_init_fn=get_worker_init_fn(self.cfg.run_cfg.seed),
                        )
                    )
                else:
                    if self.use_distributed:
                        sampler = DistributedSampler(
                            ds,
                            shuffle=True,
                            num_replicas=get_world_size(),
                            rank=get_rank(),
                            # seed
                            seed=self.cfg.run_cfg.seed + get_rank(),
                        )
                    else:
                        sampler = None
                    loader = DataLoader(
                        ds,
                        batch_size=self.cfg.run_cfg.batch_size_train,
                        num_workers=self.cfg.run_cfg.num_workers,
                        pin_memory=True,
                        sampler=sampler,
                        shuffle=sampler is None,
                        collate_fn=collate_fn,
                        # seed
                        worker_init_fn=get_worker_init_fn(self.cfg.run_cfg.seed),
                    )
                    loader = IterLoader(loader, use_distributed=self.use_distributed)
                dataloaders.append(loader)
                ratios.append(ds.sample_ratio)
            self.train_loader = MultiIterLoader(loaders=dataloaders, ratios=ratios)
        else:
            assert self.evaluate_only, "No training dataset is provided."

    def _load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Checkpoint file {ckpt_path} does not exist.")

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.accelerator.scaler and "scaler" in checkpoint:
            self.accelerator.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = checkpoint["epoch"] + 1
        logging.info(
            "Resume checkpoint from {}, starting epoch {}".format(
                ckpt_path, self.start_epoch
            )
        )

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        state_dict = self.accelerator.get_state_dict(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self._model.named_parameters()
        }
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            # "scheduler": self.lr_scheduler.state_dict(),
            "config": self.cfg.to_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if self.accelerator.scaler
            else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.accelerator.project_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
        (
            self.unwrap_dist_model(self.model).llm_tokenizer.save_pretrained(
                self.accelerator.project_dir
            ),
        )

    def _prepare_all(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)
        self.model, self.optimizer = self.accelerator.prepare(
            self._model, self.optimizer
        )
        eval_loaders = []
        for loader in self.eval_loaders:
            eval_loaders.append(self.accelerator.prepare(loader))
        self.eval_loaders = eval_loaders

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    @property
    def evaluate_only(self):
        return self.cfg.run_cfg.evaluate

    @property
    def max_epoch(self):
        return int(self.cfg.run_cfg.max_epoch)

    @property
    def resume_ckpt_path(self):
        return self.cfg.run_cfg.get("resume_ckpt_path", None)

    @property
    def use_distributed(self):
        return self.cfg.distributed

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_state = self.train_epoch(cur_epoch)
                self.log_state(state=train_state, split_name="train")
                self._save_checkpoint(cur_epoch, is_best=False)

            # evaluation phase
            logging.info("Evaluating....")

            val_log = self.eval_epoch(epoch=cur_epoch)
            if self.evaluate_only:
                if is_main_process():
                    self.log_state(val_log, split_name="val")
                break

            if val_log is not None:
                if is_main_process():
                    assert (
                        "agg_metrics" in val_log
                    ), "No agg_metrics found in validation log."

                    agg_metrics = val_log["agg_metrics"]
                    if agg_metrics > best_agg_metric:
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics
                        self._save_checkpoint(cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})
                    self.log_state(val_log, split_name="val")

            if is_dist_avail_and_initialized():
                dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def train_epoch(self, epoch):
        self.model.train()
        if not hasattr(self.train_loader, "__next__"):
            # convert to iterator if not already
            self.train_loader = iter(self.train_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=20, fmt="{avg:.6f}"))
        iters_per_epoch = self.cfg.run_cfg.iters_per_epoch
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        print_freq = 50
        header = "Train: data epoch: [{}]".format(epoch)
        for i in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
            with self.accelerator.accumulate(self.model):
                tries = 0
                while True:
                    try:
                        samples = next(self.train_loader)
                        assert isinstance(samples, dict)
                        break
                    except Exception as exc:
                        tries += 1
                        if tries > 10:
                            raise exc
                        logging.error(f"error in dataset,{exc}")
                        continue
                samples = prepare_sample(samples)
                loss = self.model(samples)
                self.accelerator.backward(loss["loss"])
                self.lr_scheduler.step(cur_epoch=epoch, cur_step=i)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accelerator.log(
                    {k: v for k, v in loss.items() if v != 0},
                    step=iters_per_epoch * epoch + i,
                )
                self.accelerator.log(
                    {"lr": self.optimizer.param_groups[0]["lr"]},
                    step=iters_per_epoch * epoch + i,
                )

                metric_logger.update(loss=loss["loss"])
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))

        return {
            k: "{:.7f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def eval_epoch(self, epoch):
        model = self.unwrap_dist_model(self.model)
        model.eval()
        print_freq = 50
        agg_metrics = 0
        metric_dict = {}

        # Build an evaluator for each task. Here you can perform an evaluation based on the incoming datasets and task_evals.
        for eval_loader, task_eval in zip(self.eval_loaders, self.task_evals):
            logging.info("")
            task_metric = task_eval(
                model, eval_loader, self.accelerator.project_dir, print_freq
            )
            # Save task metrics and additional metrics to CSV file
            csv_file_path = os.path.join(
                self.accelerator.project_dir, "all_task_metrics.csv"
            )
            append_metrics_to_csv(csv_file_path, task_metric, task_eval, epoch)

            metric_dict[task_eval.name] = task_metric
            # Record each evaluation indicator in the log
            for metric_name, metric_value in task_metric.items():
                self.accelerator.log(
                    {
                        f"eval/{task_eval.name}/{metric_name}": metric_value,
                    },
                    step=epoch,
                )
            task_avg_metric = sum(task_metric.values()) / len(task_metric)
            agg_metrics += task_avg_metric

        if is_dist_avail_and_initialized():
            dist.barrier()
        # Calculate the mean of all evaluation indicators for all val dataset
        metric_dict["agg_metrics"] = (
            (agg_metrics) / len(self.eval_loaders) if len(self.eval_loaders) != 0 else 0
        )
        self.accelerator.log({"eval_avg": metric_dict["agg_metrics"]}, step=epoch)
        return metric_dict

    @main_process
    def log_config(self):
        with open(os.path.join(self.accelerator.project_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.cfg.to_dict(), indent=4) + "\n")

    @main_process
    def log_state(self, state, split_name):
        log_state = {**{f"{split_name}_{k}": v for k, v in state.items()}}
        with open(os.path.join(self.accelerator.project_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_state) + "\n")
