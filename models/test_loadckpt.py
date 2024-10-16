from .openad.utils import build_model_checkpoint
from gorilla.config import Config
seg_point_encoder_config_path = "/workspace/project/Research_3D_Aff/Programme_affllm_code/models/openad/model/openad_pn2_modify_sam_decoder.py"
seg_point_encoder_path = "/workspace/project/Research_3D_Aff/Programme_affllm_code/model_ckpt/pointnet_best_model_decoder.t7"
openadcfg = Config.fromfile(seg_point_encoder_config_path)
seg_point_encoder = build_model_checkpoint(
    openadcfg, seg_point_encoder_path, is_eval=True
)