import os
import logging
import torch
import json
from typing import Any
from common.logger import MetricLogger, SmoothedValue
from common.registry import registry
from common.utils import get_rank


def calculate_mask_iou(mask1, mask2):
    # mask: (h, w)
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    iou = intersection / union
    return iou


def calculate_mask_iou_with_thre(mask1, mask2, thre):
    mask1 = (mask1 >= thre).to(torch.float32)
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    # print("intersection,uninon",intersection,union)
    iou = intersection / (union + 1e-6)
    iou[union == 0] += 1.0
    return iou


def calculate_precision_recall_accuracy(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    equal_points = pred == gt
    true_positives = ((pred == 1) & (gt == 1)).sum().item()
    positives_pred = pred.sum().item()
    positives_gt = gt.sum().item()
    precision = true_positives / positives_pred if positives_pred > 0 else 0
    recall = true_positives / positives_gt if positives_gt > 0 else 0
    accuracy = equal_points.float().mean().item()

    return accuracy, precision, recall


@registry.register_evaluator("affordance_acc")
class AffordanceAccEval:
    def __init__(self, name) -> None:
        self.name = name

    def eval_step(self, model, samples) -> Any:
        samples.update({"category": "ref"})
        output = model.generate(
            samples,
            num_beams=1,
            max_length=30,
        )
        answer = output["text"]
        pred_mask = output["masks"]

        iou, correct_5 = 0.0, 0.0

        pointAcc, pointPrecision, pointRecall = 0.0, 0.0, 0.0

        num = sum([gt_masks.shape[0] for gt_masks in samples["masks"]])

        for idx, (pred, gt) in enumerate(zip(pred_mask, samples["masks"])):
            n_pred = pred.shape[0]
            n_gt = gt.shape[0]

            if n_gt == 0:
                continue

            if n_pred > n_gt:
                pred = pred[:n_gt, ...]

            for mask1, mask2 in zip(pred, gt):
                if mask1.sum() == 0:
                    continue

                _iou = calculate_mask_iou(mask1, mask2)

                _pointAcc, _pointPrecision, _pointRecall = (
                    calculate_precision_recall_accuracy(pred=mask1, gt=mask2)
                )

                correct_5 += _iou > 0.5

                iou += _iou

                pointAcc += _pointAcc

                pointPrecision += _pointPrecision

                pointRecall += _pointRecall

        return (
            correct_5 / num * 100,
            iou / num * 100,
            pointAcc / num * 100,
            pointPrecision / num * 100,
            pointRecall / num * 100,
            [
                {"question": q, "pred": ans, "gt": gt}
                for ans, gt, q in zip(answer, samples["answer"], samples["question"])
            ],
        )

    def __call__(self, model, dataloader, dir, print_freq=100) -> Any:
        logging.info(f"Start evaluating on {self.name}")
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "acc5", SmoothedValue(fmt="global_acc: {global_avg:.6f}")
        )
        metric_logger.add_meter(
            "iou", SmoothedValue(fmt="global_iou: {global_avg:.6f}")
        )
        metric_logger.add_meter(
            "pointAcc", SmoothedValue(fmt="global_pointAcc: {global_avg:.6f}")
        )
        metric_logger.add_meter(
            "pointPrecision",
            SmoothedValue(fmt="global_pointPrecision: {global_avg:.6f}"),
        )
        metric_logger.add_meter(
            "pointRecall", SmoothedValue(fmt="global_pointRecall: {global_avg:.6f}")
        )

        results = []  # Initialize a dictionary to store indicator values
        # Used for debugging, stopping before certain data
        counter = 0

        for samples in metric_logger.log_every(dataloader, print_freq, self.name):
            # print("frequeu",print_freq)
            acc5, iou, pointAcc, pointPrecision, pointRecall, result = self.eval_step(
                model, samples
            )
            results.extend(result)
            metric_logger.update(
                acc5=acc5,
                iou=iou,
                pointAcc=pointAcc,
                pointPrecision=pointPrecision,
                pointRecall=pointRecall,
            )
            counter += 1

        result_dir = os.path.join(dir, self.name)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"{get_rank()}.json"), "w") as f:
            json.dump(results, f)

        metric_logger.synchronize_between_processes()
        logging.info(metric_logger.global_avg())
        return {
            "acc5_global_avg": metric_logger.acc5.global_avg,
            "iou_global_avg": metric_logger.iou.global_avg,
            "pointAcc_global_avg": metric_logger.pointAcc.global_avg,
            "pointPrecision_global_avg": metric_logger.pointPrecision.global_avg,
            "pointRecall_global_avg": metric_logger.pointRecall.global_avg,
        }
