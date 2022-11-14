import platform
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from tqdm import tqdm
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
    "max_epoch": 200,
    "initial_lr": 0.001,
    "train_batch_size": 64,
    "dataset": "custom",
    "train_resume": False,
    "set_random_seed": True,
    "l2_reg": 0.0,
}

input_size = 224
resize_size = (int(input_size * 1.46), input_size)
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
)
transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(resize_size),
                                      transforms.RandomCrop(input_size),
                                      normalize])

train_dataset = get_torch_dataloader("./data_reordered", transform=transform_train)
trainloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=0)

model = resnet18(weights=None)
n_classes = len(train_dataset.classes)
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

    return (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=config["initial_lr"], weight_decay=config["l2_reg"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=int(config["max_epoch"] * 1.0)
)

for epoch in range(config["max_epoch"]):
    model = model.to(device)
    train(epoch, "netkey")
    scheduler.step()
