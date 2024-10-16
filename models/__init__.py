import logging

# from omegaconf import OmegaConf
import torch
from common.registry import registry
from models.base_model import BaseModel
from models.aff_qwen import AffordanceQwen
from models.aff_phi import AffordancePhi
from models.aff_phi_uni3d import AffordancePhiUni3d


__all__ = ["BaseModel", "AffordanceQwen", "AffordancePhi", "AffordancePhiUni3d"]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    model_cls = registry.get_model_class(name)

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # # load preprocess
    # cfg = OmegaConf.load(model_cls.default_config_path(model_type))

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device)
