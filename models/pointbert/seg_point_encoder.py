import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from models.pointbert.dvae import Group
from models.pointbert.dvae import Encoder
from models.pointbert.logger import print_log
from collections import OrderedDict
from models.pointbert.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)

# from models.pointbert.pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
from models.pointbert import misc


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # Randomly select an initial sampling point each time
    farthest: torch.Tensor = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # 固定采样点，选择距离重心最远的点作为初始点
    # barycenter = torch.mean(xyz, dim=1)
    # dist = torch.sum((xyz - barycenter.unsqueeze(1)) ** 2, dim=-1)
    # # 选择距离重心最远的点作为初始点
    # farthest = torch.argmax(dist, dim=1)

    # print("最远点采样中的随机数据",farthest)
    # print(farthest.shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return index_points(xyz, centroids)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder without hierarchical structure"""

    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


from models.pointbert.pointnet2_utils import PointNetFeaturePropagation


class DGCNN_Propagation(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        """
        K has to be 16
        """
        # print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, bias=False),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1024, 384, kernel_size=1, bias=False),
            nn.GroupNorm(4, 384),
            nn.LeakyReLU(negative_slope=0.2),
        )

    # @staticmethod
    # def fps_downsample(coor, x, num_group):
    #     xyz = coor.transpose(1, 2).contiguous() # b, n, 3
    #     fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

    #     combined_x = torch.cat([coor, x], dim=1)

    #     new_combined_x = (
    #         pointnet2_utils.gather_operation(
    #             combined_x, fps_idx
    #         )
    #     )

    #     new_coor = new_combined_x[:, :3]
    #     new_x = new_combined_x[:, 3:]

    #     return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k, coor_q)
            assert idx.shape[1] == k
            idx_base = (
                torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1)
                * num_points_k
            )
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = (
            feature.view(batch_size, k, num_points_q, num_dims)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """coor, f : B 3 G ; B C G
        coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


class SegPointTransformer(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()
        self.config = config
        # self.args = kwargs["args"]

        self.use_max_pool = (
            config.use_max_pool
        )  # * whethet to max pool the features of different tokens

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)  # 输出是256的向量
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        # self.load_checkpoint('/chuhengshuo/Models/Ulip/pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt')

        self.propagation_2 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim]
        )
        self.propagation_1 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim]
        )
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim]
        )
        # self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        # self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        # self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(self.trans_dim, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, self.cls_dim, 1)
        # if not self.args.evaluate_3d:
        #     self.load_model_from_ckpt('./data/initialize_models/point_bert_pretrained.pt')

        # self.cls_head_finetune = nn.Sequential(
        #     nn.Linear(self.trans_dim * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.cls_dim)
        # )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith("transformer_q") and not k.startswith(
                "transformer_q.cls_head"
            ):
                base_ckpt[k[len("transformer_q.") :]] = base_ckpt[k]
            elif k.startswith("base_model"):
                base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print("missing_keys")
            print(get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            print("unexpected_keys")
            print(get_unexpected_parameters_message(incompatible.unexpected_keys))

        print(
            f"[Seg_PointTransformer] Successful Loading the ckpt from {bert_ckpt_path}"
        )

    def load_checkpoint(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            if k.startswith("module.point_encoder."):
                state_dict[k.replace("module.point_encoder.", "")] = v

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print_log("missing_keys", logger="Transformer")
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger="Transformer",
            )
        if incompatible.unexpected_keys:
            print_log("unexpected_keys", logger="Transformer")
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger="Transformer",
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            # * print successful loading
            print_log(
                "Seg_PointBERT's weights are successfully loaded from {}".format(
                    bert_ckpt_path
                ),
                logger="Transformer",
            )

    def forward(self, pts, cls_label="grasp"):
        # divide the point cloud in the same form. This is important
        B, N, C = pts.shape  # 1*2048*3
        neighborhood, center = self.group_divider(pts)
        # center的shape,1*512*3
        # 查看neighborhood的数据类型是否为double
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N,1*512*256,batch为1
        group_input_tokens = self.reduce_dim(group_input_tokens)  # 1*512*384
        # prepare cls
        cls_tokens = self.cls_token.expand(
            group_input_tokens.size(0), -1, -1
        )  # 1*1*384
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  # 1*1*384
        # add pos embedding
        pos = self.pos_embed(center)  # 1*512*384
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)  # 1*513*384
        # transformer
        feature_list = self.blocks(
            x, pos
        )  # len为3,feature[0],1*513*384,feature[1],1*513*384,feature[2],1*513*384,
        feature_list = [
            self.norm(x)[:, 1:].transpose(-1, -2).contiguous() for x in feature_list
        ]  # 1*384*512
        # cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        center_level_0 = pts.transpose(-1, -2).contiguous()  # 1*3*2048
        f_level_0 = center_level_0
        # f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)

        center_level_1 = misc.fps(pts, 512).transpose(-1, -2).contiguous()  # 1*3*512
        f_level_1 = center_level_1
        center_level_2 = misc.fps(pts, 256).transpose(-1, -2).contiguous()  # 1*3*256
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()  # 1*3*512

        # init the feature by 3nn propagation
        f_level_3 = feature_list[2]  # 1*384*512
        f_level_2 = self.propagation_2(
            center_level_2, center_level_3, f_level_2, feature_list[1]
        )  # 1*384*256
        f_level_1 = self.propagation_1(
            center_level_1, center_level_3, f_level_1, feature_list[0]
        )  # 1*384*512

        # bottom up
        # f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        # f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 = self.propagation_0(
            center_level_0, center_level_1, f_level_0, f_level_1
        )  # 1*384*2048

        # FC layers
        feat = self.bn1(self.conv1(f_level_0))
        # x = self.drop1(feat)
        # x = self.conv2(x)
        # feat =  self.bn1(self.conv1(f_level_0)).float()#1*128*2048
        # 不经过后面的drop
        x = feat
        x = x.permute(0, 2, 1)
        # x = self.drop1(feat)#1*128*2048
        # x = self.conv2(x)#1*40*2048
        # # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)#1*2048*40
        # f_level_3为1*384*512,pos为512*384
        # x为seg的特征
        point_feature = f_level_3.permute(0, 2, 1)
        pos_feature = pos[:, 1:, :]
        return x, point_feature, pos_feature
        # return x, f_level_3
        # 原来分类的point encodertransformer
        # x = self.blocks(x, pos)
        # x = self.norm(x)
        # # 修改内容，引入对最大池化的选择
        # if not self.use_max_pool:
        #     return x,pos
        # # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # # ret = self.cls_head_finetune(concat_f)
        # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1) #* (来自pointLLM)concat the cls token and max pool the features of different tokens, make it B, 1, C
        # print(concat_f.shape)
        # return concat_f
