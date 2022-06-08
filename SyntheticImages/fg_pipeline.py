import random
from collections import namedtuple
from dataclasses import dataclass, field
from queue import Queue
import tkinter as tk
from tkinter import filedialog
import os
from typing import List, Tuple, Union, Any, Optional, Iterable, Dict

import numpy as np
import cv2

INITDIR = "F:\\RadarProjekt\\Synthetische Bilder\\freigestellte Blitzer"
PIXELS = 416
MIN_HEIGHT = 11
DEBUG = False


@dataclass
class Foreground:
    """class for foreground and distractor objects"""
    name: str
    annot_class: int  # -1 == distractor object
    binaries: np.ndarray
    # bounding_box: Optional[Tuple[float, ...]] = None
    logging_info: namedtuple = namedtuple("logging_info", ["augmentation_type", "value"])


class FGExecutor:
    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.source_images: Queue[str] = Queue()
        self.prepared_foreground: Queue[Foreground] = Queue()
        self.distractor_objects: List[Foreground] = []
        self.logging_stats: Dict = {"saturation": 0, "brightness": 0}
        self.load_images()

    def load_images(self) -> None:
        files: List[str] = os.listdir(self.source_path)
        for file in files:
            if file.endswith(".png"):
                self.source_images.put(file[:-4])

    def execute(self) -> Tuple[Queue, List[Foreground]]:
        while not self.source_images.empty():
            FGPreparation(self.source_images.get(), self.source_path, self)
        if DEBUG:
            temp_queue = Queue()
            while not self.prepared_foreground.empty():
                img = self.prepared_foreground.get()
                temp_queue.put(img)
                cv2.imwrite(self.source_path + img.name + ".png", img.binaries)
            self.prepared_foreground = temp_queue
            print(self.prepared_foreground.unfinished_tasks)
        return self.prepared_foreground, self.distractor_objects


class FGPreparation:
    """scale and augment foreground and distractor objects"""

    def __init__(self, image_name: str, source_path: str, fgexecutor: FGExecutor):
        self.image_name = image_name
        self.source_path = source_path
        self.fgexecutor = fgexecutor
        self.annot_class: int = -1
        self.prepare()

    def prepare(self) -> None:
        self.determine_annot_class()
        image: np.ndarray = cv2.imread(self.source_path + self.image_name + ".png", cv2.IMREAD_UNCHANGED)
        templates: List[Foreground] = self.scale_image(image)
        self.augment_images(templates)
        for template in templates:
            if template.annot_class == -1:
                self.fgexecutor.distractor_objects.append(template)
            else:
                self.fgexecutor.prepared_foreground.put(template)

    def determine_annot_class(self) -> None:
        if "Kat1" in self.image_name:
            self.annot_class = 0
        elif "Kat2" in self.image_name:
            self.annot_class = 1
        elif "Kat3" in self.image_name:
            self.annot_class = 2
        elif "Kat5" in self.image_name:
            self.annot_class = 3

    def scale_image(self, image: np.ndarray) -> List[Foreground]:
        """decrease by 5% until threshold of MIN_HEIGHT pixel is reached"""
        foregrounds: List[Foreground] = []
        Imagesize: namedtuple = namedtuple("Imagesize", ["height", "width"])
        imagesize = Imagesize(image.shape[0], image.shape[1])

        suffix = 1
        while imagesize.height > MIN_HEIGHT and imagesize.width > MIN_HEIGHT:
            if suffix > 1000:
                print(f"check scaling ratio at picture {self.image_name}. Over 1000 iterations")
                foregrounds.clear()
                break
            # boundingbox = (
            #     imagesize.height / 2, imagesize.width / 2, imagesize.height / PIXELS, imagesize.width / PIXELS)
            foregrounds.append(Foreground(self.image_name + "_" + str(suffix), self.annot_class, image))

            imagesize = Imagesize(int(round(imagesize.height * 0.95)), int(round(imagesize.width * 0.95)))
            image = cv2.resize(image, (imagesize.width, imagesize.height), interpolation=cv2.INTER_AREA)
            suffix += 1
        return foregrounds

    def augment_images(self, templates: List[Foreground]) -> None:
        for image in templates:
            i: int = random.randint(0, 2)
            # 0 == no augmentation
            if i > 0:
                # 1 == saturation, 2 == brightness
                self.do_augment(image, i)

    def do_augment(self, image: Foreground, type: int) -> None:
        """ in/decrease saturation or brightness
        type 1 == saturation, 2 == brightness"""
        values: List[int] = [30, -30]
        im: np.ndarray = image.binaries
        value: int = values[random.randint(0, 1)]
        alpha: np.ndarray = image.binaries[:, :, 3]
        hsv: np.ndarray = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        if value > 0:
            hsv[:, :, type] = np.where(hsv[:, :, type] > 255 - value, 255,
                                       hsv[:, :, type] + value)
        else:
            hsv[:, :, type] = np.where(hsv[:, :, type] < 0 - value, 0,
                                       hsv[:, :, type] + value)
        bgr: np.ndarray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        combined: np.ndarray = np.dstack((bgr, alpha))
        image.binaries = combined

        image.logging_info.augmentation_type = "saturation" if type == 1 else "brightness"
        image.logging_info.value = value
        if type == 1:
            self.fgexecutor.logging_stats["saturation"] += 1
        else:
            self.fgexecutor.logging_stats["brightness"] += 1


if __name__ == "__main__":
    source_path = tk.filedialog.askdirectory(initialdir=INITDIR, title="Choose source folder of foreground images")
    source_path += "\\"
    save_path = tk.filedialog.askdirectory(initialdir=INITDIR,
                                           title="Choose save folder for synthesized images")
    save_path += "\\"
    if source_path != "\\" and save_path != "\\":
        FGExecutor(source_path).execute()
