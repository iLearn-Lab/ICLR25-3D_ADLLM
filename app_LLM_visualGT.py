import argparse
import torch
import os
import numpy as np
import gradio as gr
import open3d as o3d
import plotly.graph_objects as go
import time
import logging
from models import load_model_and_preprocess


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    # Select the point farthest from the center of gravity as the initial point

    # barycenter = torch.mean(xyz, dim=1)
    # dist = torch.sum((xyz - barycenter.unsqueeze(1)) ** 2, dim=-1)
    # farthest = torch.argmax(dist, dim=1)

    # Select the point farthest from the center of gravity as the initial point
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def pc_norm(pc):
    """pc: NxC, return NxC"""
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc


def change_input_method(input_method):
    if input_method == "File":
        result = [gr.update(visible=True), gr.update(visible=False)]
    elif input_method == "Object ID":
        result = [gr.update(visible=False), gr.update(visible=True)]
    return result


def create_fig(points, slider, color_strings):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    # size=1.2,
                    size=slider,
                    color=color_strings,  # Use the list of RGB strings for the marker colors
                ),
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            paper_bgcolor="rgb(255,255,255)",  # Set the background color to dark gray 50, 50, 50
        ),
    )
    return fig


def start_conversation(args):
    print("[INFO] Starting conversation...")
    logging.warning("Starting conversation...")
    while True:
        print("-" * 80)
        logging.warning("-" * 80)

        # Reset the conversation template
        # conv.reset()

        def confirm_point_cloud(point_cloud_input, answer_time, slider):
            objects = None
            data = None
            # object_id_input = object_id_input.strip()

            print("%" * 80)
            logging.warning("%" * 80)

            file = point_cloud_input.name
            print(f"Uploading file: {file}.")
            logging.warning(f"Uploading file: {file}.")
            print("%" * 80)
            logging.warning("%" * 80)

            manual_no_color = "no_color" in file

            try:
                if ".ply" in file:
                    pcd = o3d.io.read_point_cloud(file)
                    points = np.asarray(pcd.points)  # xyz
                    colors = np.asarray(pcd.colors)  # rgb, if available
                    # * if no colors actually, empty array
                    if colors.size == 0:
                        colors = None
                elif ".npy" in file:
                    data = np.load(file)
                    if data.shape[1] >= 3:
                        points = data[:, :3]
                    else:
                        raise ValueError(
                            "Input array has the wrong shape. Expected: [N, 3]. Got: {}.".format(
                                data.shape
                            )
                        )
                    colors = None if data.shape[1] < 6 else data[:, 3:6]
                else:
                    raise ValueError("Not supported data format.")
            # error
            except Exception as e:
                print(f"[ERROR] {e}")
                logging.warning(f"[ERROR] {e}")

                return None, None, answer_time, None

            if manual_no_color:
                colors = None

            # if not show_color:
            colors = None  # 消除点云颜色
            if colors is not None:
                # * if colors in range(0-1)
                if np.max(colors) <= 1:
                    color_data = np.multiply(colors, 255).astype(
                        int
                    )  # Convert float values (0-1) to integers (0-255)
                # * if colors in range(0-255)
                elif np.max(colors) <= 255:
                    color_data = colors.astype(int)
            else:
                color_data = np.zeros_like(points).astype(
                    int
                )  # Default to black color if RGB information is not available
            colors = color_data.astype(np.float32) / 255  # model input is (0-1)

            # Convert the RGB color data to a list of RGB strings in the format 'rgb(r, g, b)'
            color_strings = ["rgb({},{},{})".format(r, g, b) for r, g, b in color_data]

            fig = create_fig(points, slider, color_strings)

            points = np.concatenate((points, colors), axis=1)
            if 8192 < points.shape[0]:
                points = farthest_point_sample(points, 8192)
            point_clouds = pc_norm(points)
            point_clouds = (
                torch.from_numpy(point_clouds).unsqueeze_(0).to(torch.float32)
            )

            answer_time = 0

            return fig, answer_time, point_clouds

        with gr.Blocks() as demo:
            answer_time = gr.State(value=0)
            point_clouds = gr.State(value=None)
            conv_state = gr.State(value=None)
            gr.Markdown(
                """
                # PointCloud Visualization 👀
                """
            )
            gr.Markdown(
                """
                ### Usage:
                1. Template:Please locate the areas of the object_name with function of action_name.
                2. Knife : Please locate the areas of the Knife with function of grasp.
                """
            )
            with gr.Row():
                with gr.Column():
                    point_cloud_input = gr.File(
                        visible=True, label="Upload Point Cloud File (PLY, NPY)"
                    )
                    output = gr.Plot()
                    btn = gr.Button(value="Confirm Point Cloud")
                with gr.Column():
                    output2 = gr.Plot()
                    slider = gr.Slider(
                        minimum=0, maximum=5, value=1.2, step=0.1, label="point size"
                    )
                with gr.Column():
                    chatbot = gr.Chatbot([], elem_id="chatbot", height=560)

                    def chat(point_cloud_input, message, history, slider):
                        # message = 'The areas of the Door with function of openable'
                        history = history or []
                        # prepare LLM sample
                        file = point_cloud_input.name
                        pointdata = np.load(file)
                        raw_pointdata = np.copy(pointdata)
                        LLM_input_points = (
                            torch.Tensor(pc_norm(pointdata[:, :3])).float().cuda()
                        )
                        pointdata = (
                            torch.Tensor(pc_norm(pointdata[:, :3]))
                            .float()
                            .cuda()
                            .unsqueeze(dim=0)
                            .permute(0, 2, 1)
                        )
                        sample_affordance = {
                            "question": message,
                            "points": [LLM_input_points],
                        }
                        output_aff = model.generate(
                            sample_affordance, num_beams=1, max_length=128
                        )
                        mask = output_aff["masks"][0].squeeze(dim=1)
                        response = output_aff["text"]

                        zeros = np.zeros((2048, 3))
                        pointdata = pointdata.permute(0, 2, 1).squeeze().cpu().numpy()
                        pointdata = np.concatenate((pointdata, zeros), axis=1)
                        mask = mask.int()
                        for i in range(len(pointdata)):
                            if mask[0][i] != 0:
                                pointdata[i] = pointdata[i] + np.array(
                                    [0, 0, 0, 0, 0, 1]
                                )

                        points = pointdata[:, :3]
                        colors = pointdata[:, 3:6]
                        color_data = np.multiply(colors, 255).astype(
                            int
                        )  # Convert float values (0-1)
                        color_strings = [
                            "rgb({},{},{})".format(r, g, b) for r, g, b in color_data
                        ]
                        fig = create_fig(points, slider, color_strings)

                        points = raw_pointdata[:, :3]
                        colors = raw_pointdata[:, 3:6]
                        color_data = np.multiply(colors, 255).astype(
                            int
                        )  # Convert float values (0-1)
                        color_strings = [
                            "rgb({},{},{})".format(r, g, b) for r, g, b in color_data
                        ]
                        fig2 = create_fig(points, slider, color_strings)

                        response = response[0]
                        response = response.replace("<", "&lt;").replace(">", "&gt;")

                        history.append((message, response))
                        return history, history, fig, fig2

                    with gr.Row():
                        text_input = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter",
                            container=False,
                        )
                        run_button = gr.Button("Send")

                    run_button.click(
                        fn=chat,
                        inputs=[point_cloud_input, text_input, conv_state, slider],
                        outputs=[chatbot, conv_state, output2, output],
                    )
                    clear = gr.ClearButton(
                        [text_input, chatbot, conv_state], value="Clear"
                    )
                    # text_input.submit(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(answer_generate, [chatbot, answer_time, point_clouds, conv_state], chatbot).then(lambda x : x+1,answer_time, answer_time)
                    # clear.click(clear_conv, inputs=[chatbot, conv_state], outputs=[chatbot, answer_time], queue=False)

                btn.click(
                    confirm_point_cloud,
                    inputs=[point_cloud_input, answer_time, slider],
                    outputs=[output, answer_time, point_clouds],
                )
            # input_choice.change(change_input_method, input_choice, [point_cloud_input, object_id_input])
            # run_button.click(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(answer_generate, [chatbot, answer_time, point_clouds, conv_state], chatbot).then(lambda x : x+1, answer_time, answer_time)

        demo.queue()
        demo.launch()  # server_port=7832, share=True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="3DAFFLLM")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/workspace/project/Research_3D_Aff/3D_ADLLM/demo/demo_data",
        required=False,
    )
    parser.add_argument("--pointnum", type=int, default=2048)
    parser.add_argument(
        "--log_file",
        type=str,
        default="/workspace/project/Aff_LLM_debug/demo/serving_workdirs/serving_log.txt",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="/workspace/project/Aff_LLM_debug/demo/serving_workdirs/tmp",
    )
    # For gradio
    parser.add_argument("--port", type=int, default=7810)
    parser.add_argument(
        "--checkpoint",
        help="the dir to saved model",
        default="/workspace/project/Research_3D_Aff/Ckpts/Phi_Main/full_best.pth",
    )
    parser.add_argument("--gpu", type=str, default=None, help="Number of gpus to use")
    args = parser.parse_args()

    # * make serving dirs
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    # * add the current time for log name
    args.log_file = args.log_file.replace(
        ".txt", f"_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.txt"
    )

    logging.basicConfig(
        filename=args.log_file,
        level=logging.WARNING,  # * default gradio is info, so use warning
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.warning("-----New Run-----")
    logging.warning(f"args: {args}")

    print("-----New Run-----")
    print(f"[INFO] Args: {args}")

    # * set env variable GRADIO_TEMP_DIR to args.tmp_dir
    os.environ["GRADIO_TEMP_DIR"] = args.tmp_dir

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
    model.load_from_pretrained(args.checkpoint)
    start_conversation(args)
