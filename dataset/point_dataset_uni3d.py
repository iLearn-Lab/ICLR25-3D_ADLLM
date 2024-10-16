import torch
from torch.utils.data import Dataset
import numpy as np
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


class PointDatasetUni3D(Dataset):
    def __init__(self, ann_paths=[]):
        self.all_data = []
        for ann_path in ann_paths:
            self.all_data.extend(pkl.load(open(ann_path, "rb")))
        print(len(self.all_data))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = self.all_data[index]
        # point_encoder pc_norm preprocess data
        # for uni3d,2048*6
        full_shape_coordinate = pc_norm(data["full_shape_coordinate"])
        color_info = np.zeros((2048, 3))
        full_shape_coordinate_with_color = np.hstack(
            (full_shape_coordinate, color_info)
        )
        points = torch.tensor(full_shape_coordinate_with_color).float()
        # points = torch.tensor(pc_norm(data["full_shape_coordinate"])).float()
        original_mask = data["GT"]
        masks = torch.tensor(original_mask).float().permute(1, 0).unsqueeze(0)
        answer = data["answer"]
        shape_id = data["shape_id"]
        # Process label
        if "affordance_label" in data:
            label = data["affordance_label"]
        else:
            label = data["semantic_class"] + data["part_label"]
        return {
            "question": data["instruction"],
            "points": points,
            "answer": answer,
            "masks": masks,
            "shape_id": shape_id,
            "label": label,
        }

    def collate(self, samples):
        question = []
        points = []
        answer = []
        masks = []
        shape_id = []
        label = []

        for sample in samples:
            question.append(sample["question"])
            answer.append(sample["answer"])
            points.append(sample["points"])
            masks.append(sample["masks"])
            shape_id.append(sample["shape_id"])
            label.append(sample["label"])

        return {
            "question": question,
            "points": points,
            "answer": answer,
            "masks": masks,
            "shape_id": shape_id,
            "label": label,
        }
