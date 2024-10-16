import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import pickle as pkl


def pc_norm(pc):
    """pc: NxC, return NxC"""
    """
    pc: Nx3 array
    This functions normalizes a point cloud to fit within a unit sphere.
    It first calculates the centroid of the point cloud and then subtracts
    it from all points before scaling all points to fit within a unit sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class Point2TextDataset(Dataset):
    def __init__(self, ann_paths=[]):
        self.all_data = []
        for ann_path in ann_paths:
            self.all_data.extend(pkl.load(open(ann_path, "rb")))
        print(len(self.all_data))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = self.all_data[index]
        points = torch.tensor(pc_norm(data["full_shape_coordinate"])).float()
        answer = data["answer"]

        shape_id = data["shape_id"]
        label = data["affordance_label"]
        return {
            "question": data["instruction"],
            "points": points,
            "answer": answer,
            "shape_id": shape_id + "_" + label,
        }

    def collate(self, samples):
        question = []
        points = []
        answer = []
        shape_id = []

        for sample in samples:
            question.append(sample["question"])
            answer.append(sample["answer"])
            points.append(sample["points"])
            shape_id.append(sample["shape_id"])

        return {
            "question": question,
            "points": points,
            "answer": answer,
            "shape_id": shape_id,
        }
