import sys
import os
import random
import torch
import matplotlib.pyplot as plt
import cv2 as cv
from plantcv import plantcv as pcv

from torchvision import transforms
from PIL import Image

from utils.TinyVGG import TinyVGG
from Transformation import get_mask, get_rmask, get_roi, get_landmarks


def predict_img(model, img):
    model.eval()

    with torch.inference_mode():
        y_pred = model(img)
        y_pred_labels = torch.argmax(y_pred, dim=1)

    return y_pred_labels


def plot_prediction(img_path, pred_class, true_class):
    fig, ax = plt.subplots(1, 2)

    img = cv.imread(img_path)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ax[0].imshow(img_rgb)
    ax[0].axis("off")
    ax[0].set_title(true_class, fontweight="bold")

    trans = random.choice(
        ["Gaussian blur", "Mask", "Roi objects",
         "Analyze object", "Pseudolandmarks"]
    )
    mask = get_mask(img)
    if trans == "Gaussian blur":
        trans_img = mask
    elif trans == "Mask":
        trans_img = get_rmask(img, mask)
    elif trans == "Roi objects":
        trans_img = get_roi(img, mask)
    elif trans == "Analyze object":
        trans_img = pcv.analyze.size(img=img, labeled_mask=mask)
    elif trans == "Pseudolandmarks":
        trans_img = get_landmarks(img, mask)

    trans_img_rgb = cv.cvtColor(trans_img, cv.COLOR_BGR2RGB)
    ax[1].imshow(trans_img_rgb)
    ax[1].axis("off")
    ax[1].set_title(
        pred_class,
        fontweight="bold",
        color="green" if pred_class == true_class else "red",
    )

    plt.tight_layout()
    plt.show()


def main(model_save, img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"USING DEVICE: {device}")

    test_transform = transforms.Compose(
        [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
    )
    pil_img = Image.open(img_path)
    # unsqueeze to get the batch-like shape [1, C, H, W]
    img = test_transform(pil_img).unsqueeze(0).to(device)

    save_dict = torch.load(model_save)
    model = TinyVGG(
        input_shape=save_dict["model_input"],
        hidden_units=save_dict["model_hidden"],
        output_shape=len(save_dict["classes"]),
    ).to(device)
    model.load_state_dict(save_dict["model_state"])

    y_pred_label = predict_img(model, img)
    second_idx = img_path.rfind("/")
    first_idx = img_path.rfind("/", 0, second_idx)
    y_true_class = img_path[first_idx + 1: second_idx]

    plot_prediction(
        img_path, pred_class=save_dict["classes"][y_pred_label],
        true_class=y_true_class
    )


def parse_args():
    if len(sys.argv) < 3:
        raise ValueError("Enter the model and the image you want to use")

    if not os.path.exists(sys.argv[1]):
        raise ValueError(
            f"The entered argument '{sys.argv[1]}' does not exist")
    if not sys.argv[1].lower().endswith(".pth"):
        raise ValueError("The entered is not a valid model format")

    if not os.path.exists(sys.argv[2]):
        raise ValueError(
            f"The entered argument '{sys.argv[2]}' does not exist")
    if (
        not sys.argv[2]
        .lower()
        .endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ):
        raise ValueError("The entered image is not in a valid image format")


if __name__ == "__main__":
    try:
        parse_args()
        main(model_save=sys.argv[1], img_path=sys.argv[2])
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
