from os import path
from common.registry import registry
from omegaconf import OmegaConf
from dataset.point_dataset import PointDataset
from dataset.point_dataset_uni3d import PointDatasetUni3D
from dataset.point2Text_dataset import Point2TextDataset


@registry.register_builder("affordance")
def build_affordance_dataset(name):
    cfg_path = "configs/datasets/affordance_datasets.yaml"
    all_cfg = OmegaConf.load(cfg_path)
    cfg = all_cfg.get(name, None)
    assert cfg is not None, f"Dataset {name} not found in {cfg_path}"
    if isinstance(cfg.ann_path, str):
        cfg.ann_path = [cfg.ann_path]
    dataset = PointDataset(
        ann_paths=[path.join(registry.root, p) for p in cfg.ann_path]
    )
    return dataset


@registry.register_builder("affordance_uni3d")
def build_affordance_val_dataset(name):
    cfg_path = "configs/datasets/affordance_datasets.yaml"
    all_cfg = OmegaConf.load(cfg_path)
    cfg = all_cfg.get(name, None)
    assert cfg is not None, f"Dataset {name} not found in {cfg_path}"
    if isinstance(cfg.ann_path, str):
        cfg.ann_path = [cfg.ann_path]
    dataset = PointDatasetUni3D(
        ann_paths=[path.join(registry.root, p) for p in cfg.ann_path]
    )
    return dataset


@registry.register_builder("affordanceVQA")
def build_affordance_dataset_vqa(name):
    cfg_path = "configs/datasets/affordanceVQA_datasets.yaml"
    all_cfg = OmegaConf.load(cfg_path)
    cfg = all_cfg.get(name, None)
    assert cfg is not None, f"Dataset {name} not found in {cfg_path}"
    if isinstance(cfg.ann_path, str):
        cfg.ann_path = [cfg.ann_path]
    dataset = Point2TextDataset(
        ann_paths=[path.join(registry.root, p) for p in cfg.ann_path]
    )
    return dataset
