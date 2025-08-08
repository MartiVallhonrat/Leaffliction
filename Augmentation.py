import os
import sys
import random
import cv2 as cv
import numpy as np


def get_crop(img, height, width):
    x_from = random.randint(0, width - width // 3 - 1)
    y_from = random.randint(0, height - height // 3 - 1)
    img_crop = img[x_from: x_from + width // 3, y_from: y_from + height // 3]
    return img_crop


def get_rot(img, height, width):
    rotMat = cv.getRotationMatrix2D(
        (width // 2, height // 2), random.randint(-180, 180), 1.0
    )
    img_rotate = cv.warpAffine(img, rotMat, (width, height))
    return img_rotate


def get_shear(img, height, width):
    # get one random point for every quarter so it doesnt reverse
    src_pts = np.float32([[0, 0], [width, 0], [width // 2, height // 2]])
    dst_pts = np.float32(
        [
            [
                np.random.randint(low=0, high=width // 2),
                np.random.randint(low=0, high=height // 2),
            ],
            [
                np.random.randint(low=width // 2, high=width),
                np.random.randint(low=0, high=height // 2),
            ],
            [
                np.random.randint(low=0, high=width),
                np.random.randint(low=height // 2, high=height),
            ],
        ]
    )
    rotMat = cv.getAffineTransform(src_pts, dst_pts)
    img_shear = cv.warpAffine(img, rotMat, (width, height))
    return img_shear


def get_project(img, height, width):
    # get one random point for every quarter so it doesnt reverse
    src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_pts = np.float32(
        [
            [
                np.random.randint(low=0, high=width // 2),
                np.random.randint(low=0, high=height // 2),
            ],
            [
                np.random.randint(low=width // 2, high=width),
                np.random.randint(low=0, high=height // 2),
            ],
            [
                np.random.randint(low=0, high=width // 2),
                np.random.randint(low=height // 2, high=height),
            ],
            [
                np.random.randint(low=width // 2, high=width),
                np.random.randint(low=height // 2, high=height),
            ],
        ]
    )
    rotMat = cv.getPerspectiveTransform(src_pts, dst_pts)
    img_projective = cv.warpPerspective(img, rotMat, (width, height))
    return img_projective


def get_enhanced(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # applying CLAHE to L-channel (light channel)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limerge = cv.merge((cl, a, b))
    enhanced_img = cv.cvtColor(limerge, cv.COLOR_LAB2BGR)
    return enhanced_img


def main(image_path):
    point_idx = image_path.rfind(".")
    slash_idx = image_path.rfind("/")
    suff = image_path[point_idx:]
    img_name = image_path[slash_idx + 1: point_idx]
    fromdir_path = image_path[:slash_idx]
    savedir_path = "./dataset/images/augmented_temp"
    os.makedirs(savedir_path, exist_ok=True)

    img = cv.imread(image_path)
    (height, width) = img.shape[:2]

    # flip image
    img_flip = cv.flip(img, random.randint(0, 1))
    cv.imwrite(f"{savedir_path}/{img_name}_Flip{suff}", img_flip)
    cv.imwrite(f"{fromdir_path}/{img_name}_Flip{suff}", img_flip)

    # crop image
    img_crop = get_crop(img, height, width)
    cv.imwrite(f"{savedir_path}/{img_name}_Crop{suff}", img_crop)
    cv.imwrite(f"{fromdir_path}/{img_name}_Crop{suff}", img_crop)

    # rotate image
    img_rotate = get_rot(img, height, width)
    cv.imwrite(f"{savedir_path}/{img_name}_Rotate{suff}", img_rotate)
    cv.imwrite(f"{fromdir_path}/{img_name}_Rotate{suff}", img_rotate)

    # shear image
    img_shear = get_shear(img, height, width)
    cv.imwrite(f"{savedir_path}/{img_name}_Shear{suff}", img_shear)
    cv.imwrite(f"{fromdir_path}/{img_name}_Shear{suff}", img_shear)

    # projective image
    img_projective = get_project(img, height, width)
    cv.imwrite(f"{savedir_path}/{img_name}_Projective{suff}", img_projective)
    cv.imwrite(f"{fromdir_path}/{img_name}_Projective{suff}", img_projective)

    # enhanced image
    enhanced_img = get_enhanced(img)
    cv.imwrite(f"{savedir_path}/{img_name}_enhanced{suff}", enhanced_img)
    cv.imwrite(f"{fromdir_path}/{img_name}_enhanced{suff}", enhanced_img)


def parse_args():
    if len(sys.argv) < 2:
        raise ValueError("Enter a image as argument")
    if not os.path.exists(sys.argv[1]):
        raise ValueError("The entered argument does not exist")
    if (
        not sys.argv[1]
        .lower()
        .endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ):
        raise ValueError("The entered is not a valid image format")


if __name__ == "__main__":
    try:
        parse_args()
        main(sys.argv[1])
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
