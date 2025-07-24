"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil, os, json
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


from lerobot.scripts.rl.s_obs_processor import image_processing_temp
import tyro
import random
import matplotlib.pyplot as plt

REPO_NAME = "0313_cube_red_new"
REPO_NAME_ROOT = f"/liujinxin/mjf/lerobot/data/ur"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_PATH = "/liujinxin/dataset/ur5e"
RAW_DATASET_NAMES = [
    # "0313_cube_red_test",
    # "0313_cube_red",
    # "0313_corn_red",
    # "0313_cube_red",
    # "0314_pepper_red",
    # "0314_corn_red_random",
    # "0314_cube_red_random",
    # "0314_cup_red_random",
    # "0317_corn_red",
    # "0317_corn_blue",
    # "0317_cube_red",
    # "0317_cube_blue",
    "0306_cube_red_42",
    # "0415_long",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def gripper_clip(action):
    if action[-1] > 0.2:
        action[-1] = 1
    else:
        action[-1] = 0
    return action


def load_img(step, json_file_path):
    img_path = os.path.dirname(json_file_path) + step["rgb"]
    image = np.load(img_path)
    img_path = os.path.dirname(json_file_path) + step["wrist"]
    wrist_image = np.load(img_path)
    img_path = os.path.dirname(json_file_path) + step["imgs"]
    scene_image = np.load(img_path)
    return image, wrist_image, scene_image


def padding_init_action(data, json_file_path, dataset, index):
    image, wrist_image = load_img(data[index], json_file_path)
    init_action = [
        -0.16917512103213775,
        -0.31559707628103384,
        0.19853144723339541,
        -0.44689941080662177,
        -0.8941862128013452,
        0.02208828288125548,
        0.014968006414996177,
        0.0,
    ]
    for _ in range(3):
        dataset.add_frame(
            {
                "image": image,
                "wrist_image": wrist_image,
                "actions": init_action,
                "state": init_action,
            }
        )


def save_histogram(data, filename, xlabel="Value", ylabel="Frequency", bins=20):
    print(data[:5])
    plt.figure()
    plt.hist(data, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Histogram of {filename}")
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    # Clean up any existing dataset in the output directory
    # output_path = REPO_NAME_ROOT + "/" + REPO_NAME
    if os.path.exists(REPO_NAME_ROOT):
        shutil.rmtree(REPO_NAME_ROOT)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=REPO_NAME_ROOT,
        robot_type="ur5",
        fps=10,
        features={
            "observation.image.scene_image": {
                "dtype": "image",
                "shape": (128, 128, 3),  # (540, 960, 3), # (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.image.wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),  # (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),  # (7,),
                "names": ["action"],
            },
            # "done":{
            #     "dtype": "float32",
            #     "shape": (1,),
            #     "names": ["done"]
            # },
            "next.reward": {"dtype": "float32", "shape": (1,), "names": ["done"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    plot_data = [[], [], [], [], [], []]
    state_list = []
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset_path = os.path.join(RAW_DATASET_PATH, raw_dataset_name)
        json_files_path = []

        for dp, dn, df in os.walk(raw_dataset_path):
            for file in df:
                if file.find(".json") != -1:
                    json_files_path.append(os.path.join(dp, file))

        print(f"Found {len(json_files_path)} json files in {raw_dataset_name}")
        for json_file_path in json_files_path:
            try:
                with open(json_file_path, "r") as f:
                    data = json.load(f)
                    # tasks_set = [data[0]["subtask"]]
                    # print("task set", tasks_set)
                    # padding_init_action(data, json_file_path, dataset, 0)
                    if len(data) < 20:
                        print(
                            f"Error saving episode: {json_file_path} len of data is {len(data)}"
                        )
                        continue
                    for index in range(len(data) - 1):
                        print("index", index)
                        step = data[index]
                        # print(step['pose'].shape)
                        state = np.array(step["pose"], dtype=np.float32)
                        state = gripper_clip(state)
                        # state[:-1] = 0
                        actions = np.array(data[index + 1]["pose"], dtype=np.float32)
                        actions = gripper_clip(actions)
                        # actions[:-1] = 0
                        # print(f"actions: {actions}")
                        image, wrist_image, scene_image = load_img(step, json_file_path)

                        # image = image_processing_temp(image, "rgb")
                        wrist_image = image_processing_temp(wrist_image, "wrist")
                        scene_image = image_processing_temp(scene_image, "scene")

                        print("imgs: ", wrist_image.shape, scene_image.shape)

                        # fake
                        reward = (
                            np.float32(1.0)
                            if index > 0.8 * len(data)
                            else np.float32(0.0)
                        )

                        dataset.add_frame(
                            {
                                "observation.image.wrist_image": wrist_image,
                                "observation.image.scene_image": scene_image,
                                "action": actions,
                                "observation.state": state,
                                "next.reward": reward,
                            },
                            "pap",
                        )
                        state_list.append(state)

                    dataset.save_episode()
            except Exception as e:
                print(f"Error saving episode: {json_file_path} {e}")
                continue

    print(f"state max: {np.max(state_list, axis=0)}")
    print(f"state min: {np.min(state_list, axis=0)}")

    # plot_data
    # for index in range(len(plot_data)):
    #     save_histogram(plot_data[index], f"/liujinxin/code/tram/ArmRobot-main/dataset/raw_data/hist_{index}.png")


if __name__ == "__main__":
    tyro.cli(main)
