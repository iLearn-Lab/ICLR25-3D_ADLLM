import os
import logging
import torch
import json
from typing import Any
from common.logger import MetricLogger
from common.registry import registry
from common.utils import get_rank
import pickle as pkl


def calculate_align(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    true_positives = ((pred == 1.0) & (gt == 1.0)).sum().item()
    union = ((pred == 1.0) | (gt == 1.0)).sum().item()
    gt_positives = (gt == 1.0).sum().item()
    gt_negatives = (gt == 0.0).sum().item()
    return true_positives, union, gt_positives, gt_negatives


#  用于保存预测的结果，并将结果与OpenAD中的Over_all_data指标对齐


@registry.register_evaluator("aff_eval")
class AffordanceEval:
    def __init__(self, name) -> None:
        self.name = name

    def eval_step(self, model, samples) -> Any:
        samples.update({"category": "ref"})
        results = []
        output = model.generate(
            samples,
            num_beams=1,
            max_length=30,
        )
        answer = output["text"]
        pred_mask = output["masks"]
        for i in range(len(samples["shape_id"])):
            shape_id = samples["shape_id"][i]
            label = samples["label"][i]
            GT_masks = samples["masks"][i]
            pred_mask = output["masks"][i]
            question = samples["question"][i]
            answer = output["text"][i]
            semantic_class = samples["semantic_class"][i]
            full_shape_coordinate = samples["points"][i]
            result = {
                "shape_id": shape_id,
                "label": label,
                "pred_mask": pred_mask,
                "question": question,
                "answer": answer,
                "GT_masks": GT_masks,
                "full_shape_coordinate": full_shape_coordinate,
                "semantic_class": semantic_class,
            }
            results.append(result)

        return results

    def __call__(self, model, dataloader, dir, print_freq=100) -> Any:
        logging.info(f"Start evaluating on {self.name}")
        metric_logger = MetricLogger(delimiter="  ")
        rank_results = []
        for samples in metric_logger.log_every(dataloader, print_freq, self.name):
            results = self.eval_step(model, samples)
            for idx in range(len(results)):
                rank_results.append(results[idx])  # 使用 append 方法
        result_dir = os.path.join(dir, self.name)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"{get_rank()}.pkl"), "wb") as f:
            pkl.dump(rank_results, f)  # 保存 rank_results 而不是 results
        metric_logger.synchronize_between_processes()
        return {
            "iou_align": 0.0,
        }
