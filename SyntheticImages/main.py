import csv
from collections import namedtuple
from queue import Queue
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

from fg_pipeline import FGExecutor, Foreground
from bg_pipeline import BGExecutor, Background, bounding_box

INITDIR_FG = "F:\\RadarProjekt\\Synthetische Bilder\\freigestellte Blitzer"
# INITDIR_BG = "F:\\RadarProjekt\\Training\\Training"
INITDIR_BG = "F:\\RadarProjekt\\Synthetische Bilder\\backgrounds"
PIXELS = 416

ForegroundSize: namedtuple = namedtuple("ForegroundSize", ["height", "width"])


class CombinedImage:  # als Superklasse für background festlegen wg. iou funktion?
    def __init__(self, name: str, binaries: np.ndarray, annotations: Optional[List[Tuple[float, ...]]] = None,
                 bounding_box_corners: Optional[List[namedtuple]] = None):
        self.name = name
        self.binaries = binaries
        self.annotations = [] if annotations is None else annotations  # original txt bbox content to generate new annotation file
        self.bounding_box_corners = [] if bounding_box_corners is None else bounding_box_corners


class Executor:
    def __init__(self, foreground: Queue, background: Queue, save_path: str,
                 distractor_objects: List[Foreground]) -> None:
        self.foregrounds = foreground
        self.backgrounds = background
        self.distractor_objects = distractor_objects
        self.save_path = save_path

    def execute(self):
        while not self.foregrounds.empty():
            ImageCombinator(self.foregrounds.get(), self.backgrounds.get(), self.save_path,
                            self.distractor_objects).fuse_images()
            print(self.foregrounds.unfinished_tasks)


class ImageCombinator:
    def __init__(self, foreground: Foreground, background: Background, save_path: str,
                 distractor_objects: List[Foreground]) -> None:
        self.foreground = foreground
        self.background = background
        self.distractor_objects = distractor_objects
        self.save_path = save_path

    def fuse_images(self) -> None:
        # 1 methode für truncation, die randpixelwerte zum einfügen zurück gibt, dann muss geprüft werden ob der blitzer dort etwas verdecjt
        # 1 methode
        # position zufällig auswählen, sodass foreground objekt rein passt
        if np.random.randint(0, 10) < 2:
            # self.fuse_truncated()
            self.fuse()
        else:
            combined_image = self.fuse()
        # 50% der Fälle random element aus distractor liste rausholen -> distractor müssen bei allen 3 bildern eingefügt werden

    def fuse_truncated(self):
        pass

    def fuse(self) -> List[CombinedImage]:
        # todo: break einbauen, falls ein fg mal tatsächlich nicht eingefügt werden kann?
        while True:
            fgsize = ForegroundSize(self.foreground.binaries.shape[0], self.foreground.binaries.shape[1])
            fg_position_x_in_bg = np.random.randint(0, PIXELS - fgsize.width)
            fg_position_y_in_bg = np.random.randint(0, PIXELS - fgsize.height)
            # max 75% overlap
            if self.overlap(self.background, fg_position_x_in_bg, fg_position_y_in_bg, fgsize) < 0.75:
                # fusing auf 3 arten
                noblur: CombinedImage = self.no_blur(fg_position_x_in_bg, fg_position_y_in_bg)
                self.add_annotations(noblur, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                self.add_bboxes(noblur, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                noblur.binaries = cv2.cvtColor(noblur.binaries, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_path + noblur.name + ".jpg", noblur.binaries)
                self.save_annotation_file(noblur.annotations, noblur.name)

                # gauss: CombinedImage = self.gaussian(fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                # poiss: CombinedImage = self.poisson(fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                break

    def no_blur(self, fg_position_x_in_bg: int, fg_position_y_in_bg: int) -> CombinedImage:
        mask = self.foreground.binaries[:, :, 3]  # alpha Kanal
        mask = Image.fromarray(mask)
        # cv2 default is BGR --> conversion to RGB(A) for PIL
        bg = cv2.cvtColor(self.background.binaries, cv2.COLOR_BGR2RGB)
        fg = cv2.cvtColor(self.foreground.binaries, cv2.COLOR_BGRA2RGBA)
        bg = Image.fromarray(bg)
        fg = Image.fromarray(fg)
        bg.paste(fg, box=(fg_position_x_in_bg, fg_position_y_in_bg), mask=mask)
        # bg.save("F:\\RadarProjekt\\Synthetische Bilder\\freigestellte Blitzer\\BlitzertestPILnoblur.jpg")
        bg = np.array(bg)
        return CombinedImage(self.background.name + self.foreground.name + "_noblur", bg)

    def add_annotations(self, comb_image: CombinedImage, fg_position_x_in_bg: int, fg_position_y_in_bg: int,
                        fgsize: ForegroundSize) -> None:
        # bg annotations
        comb_image.annotations = [annot for annot in self.background.annotations]
        # fg annotations yolo format:
        # class, x (center), y (center), width, height - all relative to PIXELS
        # ex: 3, 0.543270, 0.633413, 0.125000, 0.733173
        comb_image.annotations.append(
            (self.foreground.annot_class,
             str(round((fgsize.width / 2 + fg_position_x_in_bg) / PIXELS, 6)),
             str(round((fgsize.height / 2 + fg_position_y_in_bg) / PIXELS, 6)),
             str(round(fgsize.width / PIXELS, 6)),
             str(round(fgsize.height / PIXELS, 6))))

    def add_bboxes(self, comb_image: CombinedImage, fg_position_x_in_bg: int, fg_position_y_in_bg: int,
                   fgsize: ForegroundSize) -> None:
        # bg bboxes
        comb_image.bounding_box_corners = [corners for corners in self.background.bounding_box_corners]
        # fg bboxes
        comb_image.bounding_box_corners.append(
            bounding_box(fg_position_x_in_bg, fg_position_x_in_bg + fgsize.width, fg_position_y_in_bg,
                         fg_position_y_in_bg + fgsize.height))

    def overlap(self, image: Background, position_x: int, position_y: int, size: ForegroundSize) -> float:
        # calculate overlap for all potential bboxes
        max_overlap: float = 0
        erase_x1 = position_x
        erase_x2 = position_x + size.width
        erase_y1 = position_y
        erase_y2 = position_y + size.height
        for bbox in image.bounding_box_corners:
            x1 = max(erase_x1, bbox.x1)
            x2 = min(erase_x2, bbox.x2)
            y1 = max(erase_y1, bbox.y1)
            y2 = min(erase_y2, bbox.y2)

            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
            bbox_area = (bbox.x2 - bbox.x1 + 1) * (bbox.y2 - bbox.y1 + 1)

            overlap = intersection_area / bbox_area
            max_overlap = overlap if overlap > max_overlap else max_overlap

        return max_overlap

    def save_annotation_file(self, annotations, filename):
        with open(self.save_path + filename + ".txt", mode='w', newline='') as new_annotation_file:
            writer = csv.writer(new_annotation_file, delimiter=" ")
            for row in annotations:
                writer.writerow(row)


if __name__ == "__main__":
    source_path_fg = tk.filedialog.askdirectory(initialdir=INITDIR_FG,
                                                title="Choose source folder of foreground images")
    source_path_fg += "\\"
    source_path_bg = tk.filedialog.askdirectory(initialdir=INITDIR_BG,
                                                title="Choose source folder of background images")
    source_path_bg += "\\"
    save_path = tk.filedialog.askdirectory(initialdir=INITDIR_BG,
                                           title="Choose save folder for synthetically created images")
    save_path += "\\"

    if source_path_fg != "\\" and save_path != "\\" and source_path_bg != "\\":
        foreground_images: Queue
        distractor_objects: List[Foreground]
        foreground_images, distractor_objects = FGExecutor(source_path_fg).execute()
        background_images: Queue = BGExecutor(source_path_bg).execute()
        Executor(foreground_images, background_images, save_path, distractor_objects).execute()

    # bg_test = Background("test", np.zeros((10, 10)), bounding_box_corners=[bounding_box(200, 210, 200, 210)])
    # size = ForegroundSize(20, 20)
    # ImageCombinator(Queue(), Queue(), "test", None).overlap(bg_test, 200, 200, size)
