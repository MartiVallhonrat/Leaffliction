import sys
import os
import time
import matplotlib.pyplot as plt
import torch

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.CustomTransform import CustomTransform
from utils.Subset import Subset
from utils.TinyVGG import TinyVGG
from utils.train_model import train


def plot_results(results):
    epochs = range(len(results["train_loss"]))
    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Loss plots")
    ax[0].set_xlabel("Epochs")
    ax[0].plot(epochs, results["train_loss"], label="train_loss")
    ax[0].plot(epochs, results["val_loss"], label="val_loss")
    ax[0].legend(loc="upper right")

    ax[1].set_title("Accuracity plots")
    ax[1].set_xlabel("Epochs")
    ax[1].plot(epochs, results["train_acc"], label="train_acc")
    ax[1].plot(epochs, results["val_acc"], label="val_acc")
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def save_model(full_dataset, model):
    print("Saving model...")

    dst_path = "models"
    os.makedirs(dst_path, exist_ok=True)

    slash_idx = full_dataset.root.rfind("/")
    model_name = f"{full_dataset.root[slash_idx:]}_model.pth"

    save_dict = {
        "model_state": model.state_dict(),
        "model_input": model.input_shape,
        "model_hidden": model.hidden_units,
        "classes": full_dataset.classes,
    }
    torch.save(obj=save_dict, f=f"{dst_path}/{model_name}")

    print(f"Model saved in: {dst_path + model_name}")


def main(src_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"USING DEVICE: {device}")

    full_dataset = datasets.ImageFolder(root=src_path)
    train_dataset, val_dataset = torch.utils.data.random_split(
                                    full_dataset,
                                    [0.8, 0.2])

    train_transform = transforms.Compose(
        [
            CustomTransform(),
            transforms.Resize(size=(64, 64)),
            transforms.RandAugment(),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
    )
    train_dataset = Subset(train_dataset, transform=train_transform)
    val_dataset = Subset(val_dataset, transform=val_transform)

    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    NUM_EPOCHS = 100
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, shuffle=False
    )

    model = TinyVGG(
        input_shape=3, hidden_units=10, output_shape=len(full_dataset.classes)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    start_time = time.time()
    model_results = train(
        device, model, train_loader, val_loader, optimizer, loss_fn, NUM_EPOCHS
    )
    end_time = time.time()

    print(f"Total training time: {end_time - start_time:.2f} seconds")
    plot_results(results=model_results)

    save_model(full_dataset, model)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Enter a directory as argument")
    if not os.path.isdir(sys.argv[1]):
        raise ValueError("The entered argument does not exist")
    if sys.argv[1].endswith("/"):
        sys.argv[1] = sys.argv[1][:-1]

    main(sys.argv[1])
