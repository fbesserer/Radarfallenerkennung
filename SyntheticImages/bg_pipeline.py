import csv
import random
from builtins import round
from collections import namedtuple
from dataclasses import dataclass, field
from queue import Queue
import tkinter as tk
from tkinter import filedialog
import os
from typing import List, Tuple, Union, Any, Optional, Iterable, Dict

import numpy as np
import cv2

# INITDIR = "F:\\RadarProjekt\\Training\\Training"
INITDIR = "F:\\RadarProjekt\\Synthetische Bilder\\backgrounds\\"
DEBUG = False
PIXELS = 416
NR_TOTAL_IMAGES = 10

bounding_box: namedtuple = namedtuple("bounding_box", ["x1", "x2", "y1", "y2"])


class Background:
    """class for background images"""

    def __init__(self, name: str, binaries: np.ndarray, annotations: Optional[List[Tuple[float, ...]]] = None,
                 bounding_box_corners: Optional[List[namedtuple]] = None):
        self.name = name
        self.binaries = binaries
        self.annotations = [] if annotations is None else annotations  # original txt bbox content to generate new annotation file
        self.bounding_box_corners = [] if bounding_box_corners is None else bounding_box_corners


class BGExecutor:
    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.source_images: Queue[str] = Queue()
        self.prepared_images: Queue[Background] = Queue()
        self.load_images()

    def load_images(self) -> None:
        # files: List[str] = os.listdir(self.source_path)
        # # shuffling to pick random backgrounds
        # files = sorted(files, key=lambda x: random.random())
        # files = files[:NR_TOTAL_IMAGES]
        # for file in files:
        #     if file.endswith(".jpg"):
        #         self.source_images.put(file[:-4])
        dir: List[str] = []
        with os.scandir(self.source_path) as files:
            for file in files:
                if file.name.endswith(".jpg"):
                    dir.append(file.name)
        dir = sorted(dir, key=lambda x: random.random())
        dir = dir[:NR_TOTAL_IMAGES]
        for file in dir:
            self.source_images.put(file[:-4])

    def execute(self) -> None:
        while not self.source_images.empty():
            BGPreparation(self.source_images.get(), source_path, self)
        if DEBUG:
            temp_queue = Queue()
            while not self.prepared_images.empty():
                img = self.prepared_images.get()
                temp_queue.put(img)
                cv2.imwrite(self.source_path + "/" + img.name + ".jpg", img.binaries)
            self.prepared_images = temp_queue
            print(self.prepared_images)


class BGPreparation:
    """perparing background images by random erasing"""

    def __init__(self, image_name: str, source_path: str, bgexecutor: BGExecutor):
        self.image_name = image_name
        self.source_path = source_path
        self.fgexecutor = bgexecutor
        self.prepare()

    def prepare(self):
        image: Background = Background(self.image_name,
                                       cv2.imread(self.source_path + self.image_name + ".jpg", cv2.IMREAD_UNCHANGED))
        # annotation file einlesen
        self.extract_annot_file(image)
        # erase block einfügen, 4-5 positionen werden randoml selected und falls block an keiner stelle eingefügt werden kann bild discarden
        self.random_erase(image)

    def extract_annot_file(self, image):
        with open(self.source_path + self.image_name + ".txt", mode='r') as annotation_file:
            reader = csv.reader(annotation_file, delimiter=" ")
            for row in reader:
                image.annotations.append((row[0], row[1], row[2], row[3], row[4]))
                width = int(round(float(row[3]) * PIXELS, 0))
                height = int(round(float(row[4]) * PIXELS, 0))
                x = int(round(float(row[1]) * PIXELS - (width / 2), 0))
                # x = x - (width / 2)
                # x = int(round(x, 0))
                y = int(round(float(row[2]) * PIXELS - (height / 2), 0))
                # y = y - (height / 2)
                # y = int(round(y, 0))
                # image.bounding_box_corners.append(((x, y), (x + width, y), (x, y + height), (x + width, y + height)))
                # image.bounding_box_corners.append((x, y, x + width, y + height))
                image.bounding_box_corners.append(bounding_box(x, x + width, y, y + height))

    def random_erase(self, image):
        # todo: for loop einbauen falls iou nicht größer
        size = random.randint(20, 30)
        position_x = random.randint(0, 356)
        position_y = random.randint(0, 356)
        r = np.random.randint(0, 255, (size, size * 2))
        g = np.random.randint(0, 255, (size, size * 2))
        b = np.random.randint(0, 255, (size, size * 2))
        rgb = np.dstack((r, g, b))
        if self.intersection_over_union(image, position_x, position_y, size) < 0.75:
            image.binaries[position_x:position_x + size, position_y:position_y + size * 2, :] = rgb
            cv2.imwrite(INITDIR + self.image_name + "_test.jpg", image.binaries)

    def intersection_over_union(self, image: Background, position_x: int, position_y: int, size: int) -> float:
        # max 75% overlap
        max_overlap: float = 0
        erase_x1 = position_x
        erase_x2 = position_x + size * 2
        erase_y1 = position_y
        erase_y2 = position_y + size
        for bbox in image.bounding_box_corners:
            x1 = max(erase_x1, bbox.x1)
            x2 = min(erase_x2, bbox.x2)
            y1 = max(erase_y1, bbox.y1)
            y2 = min(erase_y2, bbox.y2)

            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
            erase_area = (erase_x2 - erase_x1 + 1) * (erase_y2 - erase_y1 + 1)
            bbox_area = (bbox.x2 - bbox.x1 + 1) * (bbox.y2 - bbox.y1 + 1)
            iou = intersection_area / (erase_area + bbox_area - intersection_area)
            max_overlap = iou if iou > max_overlap else max_overlap

        return max_overlap


if __name__ == "__main__":
    source_path = tk.filedialog.askdirectory(initialdir=INITDIR, title="Choose source folder of foreground images")
    source_path += "\\"
    save_path = tk.filedialog.askdirectory(initialdir=INITDIR,
                                           title="Choose save folder for synthesized images")
    save_path += "\\"
    if source_path != "\\" and save_path != "\\":
        BGExecutor(source_path).execute()
