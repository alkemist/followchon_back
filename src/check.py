# venv/bin/python -m src.check
import pathlib
import os
import cv2
import numpy
from PIL import Image

from .helpers.file import FileHelper
from .helpers.image import ImageHelper
from .helpers.array import ArrayHelper

labels_path = './live/backup/labels'
images_path = './live/backup/images'
image_width = 1024
image_height = 768
labels = ['guinea pig', 'noisette', 'stitch']

active = False

annotations = FileHelper.list_files(labels_path, r'.*\.(txt)$').tolist()
images = FileHelper.list_files(images_path, r'.*\.(jpg|png)$').tolist()

for annotation in annotations:
    annotation_path = f"{labels_path}/{annotation}"
    name = pathlib.Path(annotation).stem
    has_errors = False

    if not os.path.exists(f"{images_path}/{name}.jpg"):
        print(f"Image don't exist", f"{images_path}/{name}.jpg")
        if active:
            os.remove(annotation_path)

    # lines = FileHelper.read_lines(annotation_path)
    # lines_ok = list()
    #
    # for line in lines:
    #     values = line.split(' ')
    #     if int(values[0]) > len(labels) - 1:
    #         print(f"Annotation error", f"{labels_path}/{annotation}")
    #         has_errors = True
    #     else:
    #         lines_ok.append(line)
    #
    # if active and has_errors:
    #     file_annotation = open(annotation_path, "w")
    #     file_annotation.write(
    #         "\n".join(lines_ok)
    #     )


for image in images:
    name = pathlib.Path(image).stem
    ext = pathlib.Path(image).suffix

    if not os.path.exists(f"{labels_path}/{name}.txt"):
        print(f"Annotation don't exist", f"{images_path}/{name}")

    image_file = cv2.imread(f"{images_path}/{image}")
    (h_img, w_img) = image_file.shape[:2]

    if h_img > image_height or w_img > image_width:
        print("Image too big", w_img, h_img, f"{images_path}/{image}")

        if active:
            cv2.imwrite(
                f"{images_path}/{image}",
                ImageHelper.resize_with_ratio(image_file, image_width, image_height)
            )

    if ext != '.jpg':
        print("Image not a jpg", f"{images_path}/{image}")

        if active:
            im = Image.open(f"{images_path}/{image}")
            rgb_im = im.convert('RGB')
            rgb_im.save(f"{images_path}/{name}.jpg")

            os.remove(f"{images_path}/{image}")
