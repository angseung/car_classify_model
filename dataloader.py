import os
import json
import shutil
from typing import Any, List, Tuple, Dict
import torch
from torch.utils import data
from torch import Tensor
import torchvision
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt


class CarDataset(data.Dataset):
    def __init__(
        self, x_tensor: Tensor, y_tensor: Tensor, base_dir: str = "./data"
    ) -> None:
        super(CarDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor

    def _getitem__(self, index: int) -> Tuple:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)


def get_json_annotation(json_file_name: str = None) -> Dict:
    with open(json_file_name, "r") as json_file:
        annotation = json.load(json_file)

    return annotation


def get_annotated_image(img: Tensor = None, label: Dict = None):
    fig = plt.figure()


def reorder_train_data_dir(base_dir: str = "./data") -> None:
    class_labels = get_json_annotation(f"{base_dir}/train/labels/meta.json")[
        "class_names"
    ]
    class_dict = {index: name for index, name in enumerate(class_labels)}
    img_list = os.listdir(f"{base_dir}/train/images")
    assert (
        len(img_list)
        == get_json_annotation(f"{base_dir}/train/labels/meta.json")["num_images"]
    )

    if not os.path.exists("./data_reordered/train"):
        for label in class_labels:
            os.makedirs(f"./data_reordered/train/{label}")

    for file_name in img_list:
        json_file_name = f"{file_name[:-4]}.json"
        label = get_json_annotation(
            f"{base_dir}/train/labels/annotations/{json_file_name}"
        )
        curr_class_name = class_dict[label["instances"][0]["category_id"]]
        src = f"{base_dir}/train/images/{file_name}"
        dst = f"./data_reordered/train/{curr_class_name}/{file_name}"
        shutil.copy(src, dst)


def get_torch_dataloader(
    base_dir: str = "./data",
    target_dir: str = "./data_reordered",
    transform: torchvision.transforms = None,
) -> torch.utils.data.DataLoader:
    if not os.path.exists(target_dir):
        reorder_train_data_dir(base_dir)

    dataloader = ImageFolder(f"{target_dir}/train", transform=transform)

    return dataloader


if __name__ == "__main__":
    a = get_torch_dataloader()
