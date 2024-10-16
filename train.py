import os
import random
import logging
import argparse
from os import path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from accelerate import Accelerator
import torch.distributed as dist
from trainer import Trainer
from common.config import LLMConfig
from common.registry import registry
from common.logger import setup_logger
from common.utils import now, get_rank, update_cfg_for_dist


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    print("rank", get_rank())
    print("rankseed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def build_model(cfg):
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    return model_cls.from_config(model_config)


def build_train_datasets(cfg) -> List[Dataset]:
    datasets = []
    for name in cfg:
        dataset_config = cfg[name]
        builder_func = registry.get_builder_func(dataset_config["type"])
        dataset = builder_func(name)
        if "sample_ratio" in dataset_config:
            dataset.sample_ratio = dataset_config.sample_ratio
        else:
            dataset.sample_ratio = (
                len(dataset) / 1e4
            )  # setting default ratio to dataset size

        logging.info(f"load dataset {name}, ratio: {dataset.sample_ratio}")
        datasets.append(dataset)
    return datasets


def build_eval_dataset(cfg):
    datasets = []
    task_evals = []
    if cfg is not None:
        for name in cfg:
            dataset_config = cfg[name]
            builder_func = registry.get_builder_func(dataset_config["type"])
            dataset = builder_func(name)
            eval_func = registry.get_evaluator_func(dataset_config["eval_type"])
            task_eval = eval_func(name)

            logging.info(f"loaded eval dataset {name}-{len(dataset)}")
            datasets.append(dataset)
            task_evals.append(task_eval)
    return datasets, task_evals


def main():
    job_id = now()
    cfg = LLMConfig(parse_args())
    cfg.device = "cuda"
    assert cfg.device == "cuda"
    setup_seeds(cfg)

    project_dir = path.join("outputs", cfg.run_cfg.output_dir, job_id)
    os.makedirs(project_dir, exist_ok=True)
    from accelerate import DistributedDataParallelKwargs

    find_unused_parameters = cfg.run_cfg.get("find_unused_parameters", False)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused_parameters
    )
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=project_dir,
        mixed_precision=cfg.run_cfg.mixed_precision,
        gradient_accumulation_steps=cfg.run_cfg.get("accum_grad_iters", 1),
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.init_trackers(project_name="tb")

    if getattr(accelerator.state, "deepspeed_plugin", None):
        from accelerate.state import AcceleratorState

        AcceleratorState().deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = cfg.run_cfg.batch_size_train

    update_cfg_for_dist(cfg)
    log_file = os.path.join(project_dir, "logging.txt")
    setup_logger(log_file)
    cfg.pretty_print()

    train_datasets = build_train_datasets(cfg.train_datasets_cfg)
    eval_datasets, task_evals = build_eval_dataset(cfg.eval_datasets_cfg)
    model = build_model(cfg)

    trainer = Trainer(
        cfg=cfg,
        job_id=job_id,
        accelerator=accelerator,
        model=model,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        task_evals=task_evals,
    )
    trainer.train()


if __name__ == "__main__":
    main()
