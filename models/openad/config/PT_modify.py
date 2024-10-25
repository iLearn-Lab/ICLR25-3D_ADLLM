seed = 1
model = dict(
    type="PT_model_mlp_512",
    num_point=2048,
    nneighbor=16,
    nblocks=4,
    transformer_dim=512,
    input_dim=3,
)

training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate="None",
    semi=False,
    rotate_type=None,
    batch_size=16,
    epoch=200,
    seed=1,
    dropout=0.5,
    numworker=2,
    gpu="4",
    workflow=dict(
        train=1,
        val=1,
        openval=1,
    ),
)
