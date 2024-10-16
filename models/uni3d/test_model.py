from Uni3D.models.uni3d import create_uni3d
import torch
# args = {
#     'npoints':10000, 
#     'num_group':512, 
#     'group_size':64, 
#     'pc_encoder_dim':512,
#     'pc_model':"eva02_base_patch14_448", 
#     'pc_feat_dim':768, 
#     'embed_dim':1024,
#     'ckpt_path':"/workspace/Model/uni3d/model.pt"
# }
class Args:
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

# 创建 Args 实例
args = Args(
    pc_model="eva02_base_patch14_448",
    pretrained_pc="",
    patch_dropout=0.,
    drop_path_rate=0.1,
    npoints=10000,
    num_group=512,
    group_size=64,
    pc_encoder_dim=512,
    pc_feat_dim=768,
    embed_dim=1024,
    ckpt_path="/workspace/Model/uni3d/model.pt"
)
import numpy as np
if __name__ == '__main__':    
    model = create_uni3d(args)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    print('loaded checkpoint {}'.format(args.ckpt_path))
    sd = checkpoint['module']
    model.load_state_dict(sd)
    model.to("cuda")
    print(model)
    data=np.load("/workspace/project/Research_3D_Aff/Programme_affllm_code_chs/demo/demo_data/Chair6374.npy")
    rgb=torch.zeros(2048, 3).unsqueeze(0)
    pc=torch.tensor(data).unsqueeze(0)
    pc = pc.to('cuda', non_blocking=True)
    rgb = rgb.to('cuda', non_blocking=True)
    feature = torch.cat((pc, rgb),dim=-1)
    fe=model.encode_pc(feature)
    print(fe)
    