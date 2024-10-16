import torch
import numpy as np
from models import load_model_and_preprocess


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


if __name__ == "__main__":
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # load model
    model = load_model_and_preprocess(
        name="aff_phi",
        model_type="phi",
        is_eval=True,
        device=device,
    )
    model.load_from_pretrained(
        "/workspace/project/Aff_LLM_debug/outputs/dgcnn_llm/3d_aff/train_dgcnn_batch_4_lr_3e-4_iteration_9220_seed_1_epoch_10_only_affdecoder_mixdata/checkpoint_7.pth"
    )
    # load data
    file = "/workspace/project/Research_3D_Aff/3D_ADLLM/demo/demo_data/Knife480.npy"
    data = np.load(file)
    if data.shape[1] >= 3:
        points = data[:, :3]
    instruction = "Please locate the areas of the Knife with function of grasp."
    input_points = torch.tensor(pc_normalize(points)).float().to(device)
    input = [input_points]
    sample_affordance = {"question": instruction, "points": input}
    output_aff = model.generate(sample_affordance, num_beams=1, max_length=128)

    print(output_aff["text"])
    print(output_aff["masks"])
    print(output_aff)
