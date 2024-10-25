import torch
import os
from models.openad.utils.weights_init import weights_init
from models.openad.model import PointTransformerSeg_512MLP

model_pool = {"PT_model_mlp_512": PointTransformerSeg_512MLP}
init_pool = {"pn2_init": weights_init}


def build_model_checkpointfromddp(cfg, checkpoint=None, is_eval=True, device="cuda"):
    if hasattr(cfg, "model"):
        model_info = cfg.model
        weights_init = model_info.get("weights_init", None)
        model_name = model_info.type
        model_cls = model_pool[model_name]
        model = model_cls(model_info)
        # num_category = len(cfg.training_cfg.train_affordance)
        # model = model_cls(model_info, num_category)
        if weights_init is not None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        if checkpoint is not None:
            _, exten = os.path.splitext(checkpoint)
            if exten == ".t7":
                state = torch.load(checkpoint)
                # model.load_state_dict(state['model_state_dict'],strict=False)
                model.load_state_dict(state, strict=False)
            elif exten == ".pth":
                check = torch.load(checkpoint)
                model.load_state_dict(check["model_state_dict"], strict=False)
            print(f"{model_name} has loaded from pretrained model {checkpoint}")
        if is_eval:
            model.eval()
        return model.to(device)
    else:
        raise ValueError("Configuration does not have model config!")
