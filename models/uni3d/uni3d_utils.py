from .Uni3D.models.uni3d import create_uni3d
import yaml
import torch

class Uni3D_Config:
    def __init__(self, pc_model, patch_dropout, pretrained_pc,drop_path_rate, npoints, num_group, group_size, pc_encoder_dim, pc_feat_dim, embed_dim, ckpt_path):
        self.pc_model = pc_model
        self.pretrained_pc = pretrained_pc
        self.patch_dropout = patch_dropout
        self.drop_path_rate = drop_path_rate
        self.npoints = npoints
        self.num_group = num_group
        self.group_size = group_size
        self.pc_encoder_dim = pc_encoder_dim
        self.pc_feat_dim = pc_feat_dim
        self.embed_dim = embed_dim
        self.ckpt_path = ckpt_path

def build_uni3d(config_path,ckpt_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    uni3d_config = Uni3D_Config(**config)
    model = create_uni3d(uni3d_config)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print('loaded checkpoint {}'.format(ckpt_path))
    sd = checkpoint['module']
    model.load_state_dict(sd)
    print(model)
    return model,uni3d_config