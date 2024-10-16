from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class PointDatasetOpenAD(Dataset):
    def __init__(self, data_dir, split, partial=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.partial = partial

        self.load_data()
        # self.affordances = self.all_data[0]["affordance"]
        return

    def load_data(self):
        self.all_data = []

        if self.partial:
            with open(opj(self.data_dir, 'partial_view_%s_data.pkl' % self.split), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                temp_data = pkl.load(f)
        for _, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]+"_"+info["semantic_class"]+"_"+info["affordance_label"]
                temp_info["question"] = info["instruction"]
                temp_info["points"] = info["full_shape_coordinate"]
                temp_info["masks"] = info["GT"]
                self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]
        text = data_dict["question"]
        datas=data_dict["points"]
        gt=data_dict["masks"]

        datas, _, _ = pc_normalize(datas)

        return datas, datas, gt, text, modelid

    def __len__(self):
        return len(self.all_data)