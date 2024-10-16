import os
import logging
import torch
import json
from typing import Any
from common.logger import MetricLogger, SmoothedValue
from common.registry import registry
from common.utils import get_rank


def calculate_mask_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    iou = intersection / union
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


def calculate_average_precision(recalls, precisions):
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = torch.maximum(precisions[i - 1], precisions[i])

    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap.item()


@registry.register_evaluator("affordance_accap")
class AffordanceAccApEval:
    def __init__(self, name) -> None:
        self.name = name
        self.iou_thresholds = torch.arange(0.5, 1.0, 0.05)

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
        ap, ap50 = 0.0, 0.0

        num = sum([gt_masks.shape[0] for gt_masks in samples["masks"]])

        for idx, (pred, gt) in enumerate(zip(pred_mask, samples["masks"])):
            n_pred = pred.shape[0]
            n_gt = gt.shape[0]

            # if n_gt == 0:
            #     continue

            if n_pred > n_gt:
                pred = pred[:n_gt, ...]

            ious = []
            for mask1, mask2 in zip(pred, gt):
                # if mask1.sum() == 0:
                #     continue
                if torch.isnan(mask1).any():
                    print("Predict Appear NaN")

                _iou = calculate_mask_iou(mask1, mask2)
                ious.append(_iou)

                _pointAcc, _pointPrecision, _pointRecall = (
                    calculate_precision_recall_accuracy(pred=mask1, gt=mask2)
                )

                correct_5 += _iou > 0.5
                iou += _iou
                pointAcc += _pointAcc
                pointPrecision += _pointPrecision
                pointRecall += _pointRecall

            # Calculate AP and AP50
            ious = torch.tensor(ious)
            precision = torch.zeros(len(self.iou_thresholds))
            recall = torch.zeros(len(self.iou_thresholds))
            for i, threshold in enumerate(self.iou_thresholds):
                tp = (ious >= threshold).sum().item()
                precision[i] = tp / n_pred if n_pred > 0 else 0
                recall[i] = tp / n_gt if n_gt > 0 else 0

            ap += calculate_average_precision(recall, precision)
            ap50 += precision[0]  # AP at threshold 0.5

        return (
            correct_5 / num * 100,
            iou / num * 100,
            pointAcc / num * 100,
            pointPrecision / num * 100,
            pointRecall / num * 100,
            ap / num * 100,
            ap50 / num * 100,
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
        metric_logger.add_meter("ap", SmoothedValue(fmt="global_ap: {global_avg:.6f}"))
        metric_logger.add_meter(
            "ap50", SmoothedValue(fmt="global_ap50: {global_avg:.6f}")
        )

        results = []
        counter = 0

        for samples in metric_logger.log_every(dataloader, print_freq, self.name):
            acc5, iou, pointAcc, pointPrecision, pointRecall, ap, ap50, result = (
                self.eval_step(model, samples)
            )
            results.extend(result)
            metric_logger.update(
                acc5=acc5,
                iou=iou,
                pointAcc=pointAcc,
                pointPrecision=pointPrecision,
                pointRecall=pointRecall,
                ap=ap,
                ap50=ap50,
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
            "ap_global_avg": metric_logger.ap.global_avg,
            "ap50_global_avg": metric_logger.ap50.global_avg,
        }
