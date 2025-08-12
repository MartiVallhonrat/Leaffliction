import argparse
import sys
import os
from plantcv import plantcv as pcv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Transformation", description="Program transforms and image"
    )

    parser.add_argument(
        "-src",
        "--source",
        type=str,
        required=True,
        help="Path to image or image directory.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=[
            "Gaussian blur",
            "Mask",
            "Roi objects",
            "Analyze object",
            "Pseudolandmarks",
            "All",
        ],
        default="All",
        help='Improvement mode, if has spaces please use "Roi objects"',
    )

    if os.path.isdir(parser.parse_known_args()[0].source):
        parser.add_argument(
            "-dst",
            "--destination",
            type=str,
            required=True,
            help="Path to save the image transformation.",
        )

    return parser.parse_args()


def get_mask(img):
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (21, 21), 0)

    leaf_mask = cv.threshold(
        img_blur, 0, 255,
        cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    threshold = cv.adaptiveThreshold(
        img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 101, 3
    )
    roi_mask = cv.bitwise_and(threshold, threshold, mask=leaf_mask)

    return roi_mask


def get_rmask(img, mask):
    mask_inv = cv.bitwise_not(mask)
    white_bg = np.full(img.shape, 255, dtype=np.uint8)

    fg = cv.bitwise_or(img, img, mask=mask)
    bg = cv.bitwise_or(white_bg, white_bg, mask=mask_inv)

    r_mask = cv.bitwise_or(fg, bg)

    return r_mask


# GET ROI ESTA MALAMENT AGAFAR EL DEL ASIER
def get_roi(img, mask):
    green = np.full(img.shape, [0, 255, 0], dtype=np.uint8)
    green_mask = cv.bitwise_or(green, green, mask=mask)
    roi = cv.bitwise_or(green_mask, img)

    # contours = cv.findContours(mask, cv.RETR_EXTERNAL,
    #               cv.CHAIN_APPROX_SIMPLE)[0]
    # largest_contour = max(contours, key=cv.contourArea)
    # x, y, w, h = cv.boundingRect(largest_contour)
    # roi = cv.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return roi


def get_landmarks(img):
    # DEVOLVER PSEUDOLANDMARKS
    return img


def transform_dir(args):
    src_path = args.source
    dst_path = args.destination
    os.makedirs(dst_path, exist_ok=True)
    mode = args.mode

    for image_path in os.listdir(src_path):
        point_idx = image_path.rfind(".")
        suff = image_path[point_idx:]
        img_name = image_path[:point_idx]

        img = cv.imread(f"{src_path}/{image_path}")

        mask = get_mask(img)
        if mode == "Gaussian blur" or mode == "All":
            cv.imwrite(f"{dst_path}/{img_name}_gblur{suff}", mask)

        if mode == "Mask" or mode == "All":
            r_mask = get_rmask(img, mask)
            cv.imwrite(f"{dst_path}/{img_name}_mask{suff}", r_mask)

        if mode == "Roi objects" or mode == "All":
            roi = get_roi(img, mask)
            cv.imwrite(f"{dst_path}/{img_name}_roi{suff}", roi)

        if mode == "Analyze object" or mode == "All":
            analysis = pcv.analyze.size(img=img, labeled_mask=mask)
            cv.imwrite(f"{dst_path}/{img_name}_analysis{suff}", analysis)

        if mode == "Pseudolandmarks" or mode == "All":
            landmarks = get_landmarks(img)
            cv.imwrite(f"{dst_path}/{img_name}_landmarks{suff}", landmarks)


def multi_graph(img):
    fig, ax = plt.subplots(3, 2)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ax[0, 0].imshow(img_rgb)
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Original")

    mask = get_mask(img)
    img_rgb = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    ax[0, 1].imshow(img_rgb)
    ax[0, 1].axis("off")
    ax[0, 1].set_title("Gaussian Blur")

    img_rgb = cv.cvtColor(get_rmask(img, mask), cv.COLOR_BGR2RGB)
    ax[1, 0].imshow(img_rgb)
    ax[1, 0].axis("off")
    ax[1, 0].set_title("Mask")

    img_rgb = cv.cvtColor(get_roi(img, mask), cv.COLOR_BGR2RGB)
    ax[1, 1].imshow(img_rgb)
    ax[1, 1].axis("off")
    ax[1, 1].set_title("Roi Objects")

    img_rgb = cv.cvtColor(
        pcv.analyze.size(img=img, labeled_mask=mask), cv.COLOR_BGR2RGB
    )
    ax[2, 0].imshow(img_rgb)
    ax[2, 0].axis("off")
    ax[2, 0].set_title("Analyze Object")

    img_rgb = cv.cvtColor(get_landmarks(img), cv.COLOR_BGR2RGB)
    ax[2, 1].imshow(img_rgb)
    ax[2, 1].axis("off")
    ax[2, 1].set_title("Pseudolandmarks")


def single_graph(img, mode):
    fig, ax = plt.subplots(1, 2)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ax[0].imshow(img_rgb)
    ax[0].axis("off")
    ax[0].set_title("Original")

    mask = get_mask(img)
    if mode == "Gaussian blur":
        img_rgb = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
        ax[1].set_title("Gaussian Blur")
    elif mode == "Mask":
        img_rgb = cv.cvtColor(get_rmask(img, mask), cv.COLOR_BGR2RGB)
        ax[1].set_title("Mask")
    elif mode == "Roi objects":
        img_rgb = cv.cvtColor(get_roi(img, mask), cv.COLOR_BGR2RGB)
        ax[1].set_title("Roi Objects")
    elif mode == "Analyze object":
        img_rgb = cv.cvtColor(
            pcv.analyze.size(img=img, labeled_mask=mask), cv.COLOR_BGR2RGB
        )
        ax[1].set_title("Analyze Object")
    elif mode == "Pseudolandmarks":
        img_rgb = cv.cvtColor(get_landmarks(img), cv.COLOR_BGR2RGB)
        ax[1].set_title("Pseudolandmarks")

    ax[1].axis("off")
    ax[1].imshow(img_rgb)


def transform_graph(args):
    img = cv.imread(args.source)

    if args.mode == "All":
        multi_graph(img)
    else:
        single_graph(img, mode=args.mode)

    plt.tight_layout()
    plt.show()


def main(args):
    if os.path.isdir(args.source):
        transform_dir(args)
    else:
        transform_graph(args)


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args)
    except Exception as e:
        sys.stderr.write(f"Error: {e}")
