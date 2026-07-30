#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import re

import cv2
import numpy as np


IMAGE_EXT = (".jpg", ".jpeg", ".webp", ".bmp", ".png", ".tif", ".tiff")


def natural_key(path):
    basename = osp.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", basename)]


def get_image_list(path):
    if osp.isfile(path):
        ext = osp.splitext(path)[1].lower()
        return [path] if ext in IMAGE_EXT else []

    image_names = []
    for main_dir, _, file_names in os.walk(path):
        for filename in file_names:
            image_path = osp.join(main_dir, filename)
            if osp.splitext(image_path)[1].lower() in IMAGE_EXT:
                image_names.append(image_path)
    return sorted(image_names, key=natural_key)


def read_image(path):
    extension = osp.splitext(path)[1].lower()
    if extension in (".tif", ".tiff"):
        image = _read_tiff(path)
        if image is not None:
            return image
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image

    try:
        from PIL import Image

        with Image.open(path) as pil_image:
            converted = pil_image.convert("RGB")
            return cv2.cvtColor(np.asarray(converted), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def write_image(path, image):
    extension = osp.splitext(path)[1] or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    encoded.tofile(path)
    return True


def _read_tiff(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint16:
        image = (image / 257).clip(0, 255).astype(np.uint8)
    return image
