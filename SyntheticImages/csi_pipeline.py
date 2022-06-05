import queue
import random
from collections import namedtuple
from dataclasses import dataclass
from queue import Queue
import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import askyesno
import os
from typing import List, Tuple, Union, Any, Optional, Iterable

import numpy as np
import cv2

INITDIR = "F:\\RadarProjekt\\Synthetische Bilder\\freigestellte Blitzer"
PIXELS = 416
MIN_HEIGHT = 20


@dataclass
class Foreground:
    """class for foreground and distractor objects"""
    name: str
    annot_class: int  # 0 == distractor object
    binaries: np.ndarray
    bounding_box: Optional[Tuple[float, ...]] = None
    # def __init__(self, name: str, annot_class: int = 0, binaries: np.ndarray = None,
    #              bounding_box: Optional[Tuple[float, ...]] = None) -> None:
    #     # annot_class = 0 for distractor objects
    #     self.name = name
    #     self.annot_class = annot_class
    #     self.binaries = binaries
    #     self.bounding_box = () if bounding_box is None else bounding_box
    #     # bounding box x,y werte müssen später noch je nachdem wo im bild platziert angepasst werden (tatsächliche position + x/y)


class FGExecutor:
    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.source_images: Queue = Queue()
        self.prepared_images: Queue = Queue()
        self.load_images()

    def load_images(self) -> None:
        files: List[str] = os.listdir(self.source_path)
        for file in files:
            if file.endswith(".png"):
                self.source_images.put(file[:-4])

    def execute(self) -> None:
        while not self.source_images.empty():
            FGPreparation(self.source_images.get(), source_path, self)


class FGPreparation:
    def __init__(self, image_name: str, source_path: str, fgexecutor: FGExecutor):
        self.image_name = image_name
        self.source_path = source_path
        self.fgexecutor = fgexecutor
        self.annot_class: int = 0
        self.prepare()

    def prepare(self) -> None:
        self.determine_annot_class()
        image: np.ndarray = cv2.imread(self.source_path + self.image_name + ".png", cv2.IMREAD_UNCHANGED)
        # templates erstellen
        templates: List[Foreground] = self.scale_image(image)
        # templates augmentieren
        self.augment_images(templates)
        # fertige templates in die prepared images schlange einreihen
        # self.fgexecutor.prepared_images.put(foreground)

    def determine_annot_class(self):
        if "Kat1" in self.image_name:
            self.annot_class = 1
        elif "Kat2" in self.image_name:
            self.annot_class = 2
        elif "Kat3" in self.image_name:
            self.annot_class = 3
        elif "Kat5" in self.image_name:
            self.annot_class = 5

    def scale_image(self, image: np.ndarray) -> List[Foreground]:
        # um 25% kleiner machen bis eine Schwelle von MIN_HEIGHT Pixel erreicht ist
        foregrounds: List[Foreground] = []
        Imagesize: namedtuple = namedtuple("Imagesize", ["height", "width"])
        imagesize = Imagesize(image.shape[0], image.shape[1])

        # boundingbox = (imagesize.height / 2, imagesize.width / 2, imagesize.height / PIXELS, imagesize.width / PIXELS)
        # foregrounds.append(Foreground(self.image_name + "_original", self.annot_class, image, boundingbox))
        suffix = 1
        while imagesize.height > MIN_HEIGHT and imagesize.width > MIN_HEIGHT // 2:
            boundingbox = (
                imagesize.height / 2, imagesize.width / 2, imagesize.height / PIXELS, imagesize.width / PIXELS)
            foregrounds.append(Foreground(self.image_name + "_" + str(suffix), self.annot_class, image, boundingbox))

            imagesize = Imagesize(int(imagesize.height * 0.75), int(imagesize.width * 0.75))
            image = cv2.resize(image, (imagesize.width, imagesize.height), interpolation=cv2.INTER_AREA)
            suffix += 1
        return foregrounds

    def augment_images(self, templates: List[Foreground]) -> None:
        for image in templates:
            i: int = random.randint(0, 2)
            # 0 == no augmentation
            if i == 1:
                self.change_brightness(image)
            elif i == 2:
                self.change_saturation(image)

    def change_brightness(self, image: Foreground):
        values = [30, -30]
        im = image.binaries
        value = values[random.randint(0, 1)]
        # todo: 4. dimension speichern und nachher wieder anfügen
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        if value > 0:
            hsv[:, :, 2] = np.where(hsv[:, :, 2] > 255 - value, 255,
                                    hsv[:, :, 2] + value)
        else:
            hsv[:, :, 2] = np.where(hsv[:, :, 2] < 0 - value, 0,
                                    hsv[:, :, 2] + value)
        image.binaries = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def change_saturation(self, image: Foreground):
        pass


if __name__ == "__main__":
    source_path = tk.filedialog.askdirectory(initialdir=INITDIR, title="Choose source folder of foreground images")
    source_path += "\\"
    save_path = tk.filedialog.askdirectory(initialdir=INITDIR,
                                           title="Choose save folder for synthesized images")
    save_path += "\\"
    if source_path != "\\" and save_path != "\\":
        FGExecutor(source_path).execute()
        # FGExecutor.execute(FGExecutor(source_path))
        # ex = FGExecutor(source_path)
        # ex.execute()
