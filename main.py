import os
import platform
import json
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, resnet50
from torchvision.io import read_image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchinfo import summary
from tqdm import tqdm
from PIL import Image
from dataloader import get_torch_dataloader

curr_os = platform.system()
print("Current OS : %s" % curr_os)

if "Windows" in curr_os:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif "Darwin" in curr_os:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif "Linux" in curr_os:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    "model" : "resnet18",
    "max_epoch": 100,
    "initial_lr": 0.0015,
    "train_batch_size": 16,
    "dataset": "custom",
    "train_resume": False,
    "set_random_seed": True,
    "l2_reg": 0.00075,
}

best_acc = 0
input_size = 256
# resize_size = (int(input_size * 1.46), input_size)
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
)
transform_train = transforms.Compose([transforms.ToTensor(),
                                    #   transforms.Resize(resize_size),
                                      transforms.RandomCrop(input_size),
                                      normalize])

transform_test = transforms.Compose([transforms.ToTensor(),
                                    #   transforms.Resize(resize_size),
                                      transforms.CenterCrop(input_size),
                                      normalize])

dataset = get_torch_dataloader(base_dir="./data", target_dir="./data_reordered", transform=transform_train)
train_dataset, test_dataset = random_split(dataset, [500, 100])

trainloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=0)

model = resnet50(weights=None)
n_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, n_classes)


# Training
def train(epoch, dir_path=None, plotter=None) -> None:
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    with tqdm(trainloader, unit="batch") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Train Epoch {epoch}")

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(
                loss=train_loss / (batch_idx + 1), accuracy=100.0 * correct / total
            )

    with open("outputs/" + dir_path + "/log.txt", "a") as f:
        f.write(
            "Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
        )


def test(epoch, dir_path=None, plotter=None) -> None:
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    if dir_path is None:
        dir_path = "outputs/checkpoint"
    else:
        dir_path = "outputs/" + dir_path

    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Test Epoch {epoch}")

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(
                    loss=test_loss / (batch_idx + 1), accuracy=100.0 * correct / total
                )
    acc = 100.0 * correct / total

    # Save checkpoint.
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, "./" + dir_path + "/ckpt.pth")

        best_acc = acc

    with open(dir_path + "/log.txt", "a") as f:
        f.write("|Test| Loss: %.3f, Acc: %.3f \n" % (test_loss / (batch_idx + 1), acc))


def save_model(dir_path: str = None) -> None:
    print("Saving..")
    state = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        # "acc": acc,
        "epoch": epoch,
    }
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    torch.save(state, "./" + dir_path + "/ckpt.pth")


def model_inference(model, test_dir: str = "./data/test") -> None:
    model = model.to(device)
    img_list = os.listdir(test_dir)

    for fname in img_list:
        img = Image.open(f"{test_dir}/{fname}")
        img= transform_test(img)[None, :, :, :].to(device)
        inference = model(img)

        print(torch.argmax(inference, axis=1))

    return inference


start_epoch = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=config["initial_lr"], weight_decay=config["l2_reg"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=int(config["max_epoch"] * 1.0)
)

model_name = config["model"]
if not os.path.exists(f"./outputs/{model_name}"):
    os.makedirs(f"./outputs/{model_name}")

test_dir = "./data/test"

with open(f"./outputs/{model_name}/log.txt", "w") as f:
    f.write(f"Networks : {model_name}\n")
    f.write(f"Net Train Configs: \n {json.dumps(config)}\n")
    m_info = summary(model, (1, 3, input_size, input_size), verbose=0)
    f.write(f"{str(m_info)}\n")

if config["train_resume"]:
    # Load checkpoint.
    checkpoint = torch.load(f"outputs/{model_name}/ckpt.pth")
    model.load_state_dict(checkpoint["net"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"==> Resuming from checkpoint..., epoch: {start_epoch}")

for epoch in range(start_epoch, config["max_epoch"]):
    model = model.to(device)
    train(epoch, model_name)
    test(epoch, model_name)
    # save_model(f"./outputs/{model_name}")
    scheduler.step()

model_inference(model, test_dir=test_dir)
