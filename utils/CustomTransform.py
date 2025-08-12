import random
import os
import cv2 as cv
import numpy as np
from plantcv import plantcv as pcv

from Transformation import get_mask, get_rmask, get_roi, get_landmarks


class CustomTransform:
    def __init__(self):
        self._num = 0

    def __call__(self, input_img):
        # Transform PIL Image into OpenCV format
        img = np.array(input_img)

        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        trans = random.choice(
            [
                "Gaussian blur",
                "Mask",
                "Roi objects",
                "Analyze object",
                "Pseudolandmarks",
            ]
        )
        os.makedirs("transformed_train", exist_ok=True)

        # Apply transform here
        mask = get_mask(img_bgr)
        if trans == "Gaussian blur":
            trans_img = mask
        elif trans == "Mask":
            trans_img = get_rmask(img_bgr, mask)
        elif trans == "Roi objects":
            trans_img = get_roi(img_bgr, mask)
        elif trans == "Analyze object":
            trans_img = pcv.analyze.size(img=img_bgr, labeled_mask=mask)
        elif trans == "Pseudolandmarks":
            trans_img = get_landmarks(img_bgr)

        # Save transformed image in an extra folder here
        cv.imwrite(f"transformed_train/image ({self._num}).JPG", trans_img)
        self._num = self._num + 1

        # Return transformed image
        # trans_img_rgb = cv.cvtColor(trans_img, cv.COLOR_BGR2RGB)
        # trans_img_pil = Image.fromarray(trans_img_rgb)

        return input_img
