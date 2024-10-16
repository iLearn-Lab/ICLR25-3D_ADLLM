from typing import List, Tuple, Type
import torch
from torch import nn


class AffDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 1,
    ) -> None:
        """
        Predicts masks given an point cloud and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

    def forward(
        self,
        pointcloud_embeddings: torch.Tensor,
        pointcloud_emorigin: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given point cloud and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the PointBackbone
          sparse_prompt_embeddings (torch.Tensor): the embeddings from AFF token
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        masks = self.predict_masks(
            pointcloud_embeddings=pointcloud_embeddings,
            pointcloud_emorigin=pointcloud_emorigin,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :]
        return masks

    def predict_masks(
        self,
        pointcloud_embeddings: torch.Tensor,
        pointcloud_emorigin: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = pointcloud_embeddings
        b, N, c = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pointcloud_emorigin, tokens)  # src,b*2048*512
        mask_tokens_out = hs[:, 0:1, :]

        # predict affordance
        src = src.transpose(1, 2)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(mask_tokens_out[:, i, :])
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, N = src.shape
        masks = (hyper_in @ src.view(b, c, N)).view(b, -1, N)

        # Generate mask quality predictions
        return masks

    def counting_training_parameters(self):
        total = 0.0
        trainable_names = []
        all = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total += param.nelement()
                trainable_names.append(name)
            all += param.nelement()
        print(trainable_names)
        print("  + Number of trainable params: %.2fM" % (total / 1e6))
        print("Number of all params: %.2fM" % (all / 1e6))
        return total
