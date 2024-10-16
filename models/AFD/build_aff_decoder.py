import torch
from .modeling import (AffDecoder,TwoWayTransformer)

def build_affdecoder_output_512_mlp(checkpoint=None):
    return _build_aff_output_mlp512_dim(checkpoint=checkpoint)

build_AFD = build_affdecoder_output_512_mlp

aff_decoder_model_registry = {
    "aff_decoder":build_affdecoder_output_512_mlp,
}
def _build_aff_output_mlp512_dim(checkpoint=None):
    prompt_embed_dim = 32
    Aff=AffDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=32,
                mlp_dim=512,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
        )
    Aff.eval()
    if checkpoint is not None:
        ckpt_model = torch.load(checkpoint, map_location='cpu')
        if "model" in ckpt_model.keys():
            state_dict = ckpt_model["model"]
        elif "model_state_dict" in ckpt_model.keys():
            state_dict = ckpt_model["model_state_dict"]
        else:
            state_dict = ckpt_model
        state_dict = {k.replace('aff_model.', ''): v for k, v in state_dict.items() if k.startswith('aff_model.')} 
        print("\nLoad AFF Model Successfully")
        incompatible_keys = Aff.load_state_dict(state_dict, strict=False)
        print("AffDecoder_OpenAD_Output_one incompatible keys: ", incompatible_keys)

    return Aff