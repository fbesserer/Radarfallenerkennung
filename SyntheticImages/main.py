import csv
import math
from collections import namedtuple
from queue import Queue
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

from fg_pipeline import FGExecutor, Foreground
from bg_pipeline import BGExecutor, Background, bounding_box

INITDIR_FG = "F:\\RadarProjekt\\Synthetische Bilder\\freigestellte Objekte"
# INITDIR_BG = "F:\\RadarProjekt\\Training\\Training"
INITDIR_BG = r"F:\RadarProjekt\Training\Training"
INITDIR_SAVE = "F:\\RadarProjekt\\Synthetische Bilder\\generatedPics"
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
        i: int = 0
        while not self.foregrounds.empty():
            ImageCombinator(self.foregrounds.get(), self.backgrounds.get(), self.save_path,
                            self.distractor_objects).fuse_images()
            print(i)
            i += 1
            # print(f" foreground unfinished tasks: {self.foregrounds.unfinished_tasks}")
            # print(f" background unfinished tasks: {self.backgrounds.unfinished_tasks}")


class ImageCombinator:
    def __init__(self, foreground: Foreground, background: Background, save_path: str,
                 distractor_objects: List[Foreground]) -> None:
        self.foreground = foreground
        self.background = background
        self.distractor_objects = distractor_objects
        self.save_path = save_path

    def fuse_images(self) -> None:
        # if np.random.randint(0, 10) < 2:
        #     self.truncate_foreground()
        combined_images: List[CombinedImage] = self.combine_fg_with_bg()

        # insert distractor in 50% of all cases
        if np.random.randint(0, 10) < 5:
            self.add_distractor_object(combined_images)

        for image in combined_images:
            # save images and create annotation files
            cv2.imwrite(self.save_path + image.name + ".jpg", image.binaries)
            self.save_annotation_file(image.annotations, image.name)

    def truncate_foreground(self) -> None:
        # rectangle size: a, b
        x_axis: int = self.foreground.binaries.shape[1]
        y_axis: int = self.foreground.binaries.shape[0]
        min_area: int = int(round((y_axis * x_axis) / 2, 0))  # 50% of area
        x1: int = np.random.randint(0, x_axis / 2)
        a: int = x_axis - x1
        y_min: int = int(round(min_area / a, 0))
        y1: int = np.random.randint(0, y_axis - y_min + 1)
        b: int = y_axis - y1

        assert a * b >= min_area

        x2: int = np.random.randint(math.ceil(min_area / b) + x1, x_axis + 1)
        a = x2 - x1
        y2 = np.random.randint(math.ceil(min_area / a) + y1, y_axis + 1)
        b = y2 - y1

        assert a * b >= min_area

        self.foreground.binaries = self.foreground.binaries[y1:y2, x1:x2]

    def combine_fg_with_bg(self, truncate: bool = False) -> List[CombinedImage]:
        # todo: break einbauen, falls ein fg mal tatsächlich nicht eingefügt werden kann? - probiert 1000 mal
        images: List[CombinedImage] = []
        tries = 0
        while True and tries < 1000:
            fgsize: ForegroundSize = ForegroundSize(self.foreground.binaries.shape[0],
                                                    self.foreground.binaries.shape[1])
            if truncate:  # todo: implement truncate functionality
                fg_position_x_in_bg: int = 0
                fg_position_y_in_bg: int = 0

            else:
                fg_position_x_in_bg: int = np.random.randint(0, PIXELS - fgsize.width)
                fg_position_y_in_bg: int = np.random.randint(0, PIXELS - fgsize.height)

            # max 75% overlap with existing radar trap
            if self.overlap(self.background, fg_position_x_in_bg, fg_position_y_in_bg, fgsize) < 0.75:
                # 3 different fusing strategies are individually applied
                noblur: CombinedImage = self.no_blur(fg_position_x_in_bg, fg_position_y_in_bg)
                self.combine_annotations(noblur, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                self.combine_bboxes(noblur, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                noblur.binaries = cv2.cvtColor(noblur.binaries, cv2.COLOR_RGB2BGR)

                gauss: CombinedImage = self.gaussian(fg_position_x_in_bg, fg_position_y_in_bg)
                self.combine_annotations(gauss, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                self.combine_bboxes(gauss, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                gauss.binaries = cv2.cvtColor(gauss.binaries, cv2.COLOR_RGB2BGR)

                poiss: CombinedImage = self.poisson(fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                self.combine_annotations(poiss, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                self.combine_bboxes(poiss, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                poiss.binaries = cv2.cvtColor(poiss.binaries, cv2.COLOR_RGB2BGR)

                images.extend([noblur, gauss, poiss])
                break

            tries += 1
        return images

    def no_blur(self, fg_position_x_in_bg: int, fg_position_y_in_bg: int) -> CombinedImage:
        mask: Image = Image.fromarray(self.foreground.binaries[:, :, 3])  # alpha Kanal

        # cv2 default is BGR --> conversion to RGB(A) for PIL
        bg: Image = Image.fromarray(cv2.cvtColor(self.background.binaries, cv2.COLOR_BGR2RGB))
        fg: Image = Image.fromarray(cv2.cvtColor(self.foreground.binaries, cv2.COLOR_BGRA2RGBA))

        bg.paste(fg, box=(fg_position_x_in_bg, fg_position_y_in_bg), mask=mask)

        return CombinedImage(self.background.name + self.foreground.name + "_noblur", np.array(bg))

    def gaussian(self, fg_position_x_in_bg: int, fg_position_y_in_bg: int) -> CombinedImage:
        mask_gauss: np.ndarray = cv2.GaussianBlur(self.foreground.binaries[:, :, 3], (5, 5), 2)  # alpha Kanal
        mask: Image = Image.fromarray(mask_gauss)

        # cv2 default is BGR --> conversion to RGB(A) for PIL
        bg: Image = Image.fromarray(cv2.cvtColor(self.background.binaries, cv2.COLOR_BGR2RGB))
        fg: Image = Image.fromarray(cv2.cvtColor(self.foreground.binaries, cv2.COLOR_BGRA2RGBA))
        bg.paste(fg, box=(fg_position_x_in_bg, fg_position_y_in_bg), mask=mask)

        return CombinedImage(self.background.name + self.foreground.name + "_gauss", np.array(bg))

    def poisson(self, fg_position_x_in_bg: int, fg_position_y_in_bg: int, fgsize: ForegroundSize) -> CombinedImage:
        # completely white mask keeps more details in foreground
        mask_poisson: np.ndarray = np.full((self.foreground.binaries.shape[0], self.foreground.binaries.shape[1]),
                                           255).astype("uint8")
        fg: np.ndarray = cv2.cvtColor(self.foreground.binaries[:, :, 0:3], cv2.COLOR_BGR2RGB)
        bg: np.ndarray = cv2.cvtColor(self.background.binaries, cv2.COLOR_BGR2RGB)

        center: tuple = (fg_position_x_in_bg + fgsize.width // 2, fg_position_y_in_bg + fgsize.height // 2)
        new: np.ndarray = cv2.seamlessClone(fg, bg, mask_poisson, center, cv2.NORMAL_CLONE)
        return CombinedImage(self.background.name + self.foreground.name + "_poisson", new)

    def combine_annotations(self, comb_image: CombinedImage, fg_position_x_in_bg: int, fg_position_y_in_bg: int,
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

    def combine_bboxes(self, comb_image: CombinedImage, fg_position_x_in_bg: int, fg_position_y_in_bg: int,
                       fgsize: ForegroundSize) -> None:
        # bg bboxes
        comb_image.bounding_box_corners = [corners for corners in self.background.bounding_box_corners]
        # fg bboxes
        comb_image.bounding_box_corners.append(
            bounding_box(fg_position_x_in_bg, fg_position_x_in_bg + fgsize.width, fg_position_y_in_bg,
                         fg_position_y_in_bg + fgsize.height))

    def add_distractor_object(self, combined_images: List[CombinedImage]) -> None:
        distractor: Foreground = self.distractor_objects[np.random.randint(0, len(self.distractor_objects))]
        distractor_image = distractor.binaries.copy()
        assert distractor_image is not distractor.binaries
        tries: int = 0
        noblur, gauss, poisson = combined_images
        while True:
            tries += 1
            fgsize: ForegroundSize = ForegroundSize(distractor_image.shape[0], distractor_image.shape[1])
            fg_position_x_in_bg: int = np.random.randint(0, PIXELS - fgsize.width)
            fg_position_y_in_bg: int = np.random.randint(0, PIXELS - fgsize.height)
            # no overlap
            if self.overlap(noblur, fg_position_x_in_bg, fg_position_y_in_bg, fgsize) == 0:
                # blending zufällig auswählen
                i = np.random.randint(0, 3)
                # no blur
                if i == 0:
                    self.distractor_nonblured(distractor_image, noblur, fg_position_x_in_bg, fg_position_y_in_bg)
                    self.distractor_nonblured(distractor_image, gauss, fg_position_x_in_bg, fg_position_y_in_bg)
                    self.distractor_nonblured(distractor_image, poisson, fg_position_x_in_bg, fg_position_y_in_bg)
                # gauss
                elif i == 2:
                    self.distractor_gauss(distractor_image, noblur, fg_position_x_in_bg, fg_position_y_in_bg)
                    self.distractor_gauss(distractor_image, gauss, fg_position_x_in_bg, fg_position_y_in_bg)
                    self.distractor_gauss(distractor_image, poisson, fg_position_x_in_bg, fg_position_y_in_bg)

                # poisson
                elif i == 3:
                    self.distractor_poisson(distractor_image, noblur, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                    self.distractor_poisson(distractor_image, gauss, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)
                    self.distractor_poisson(distractor_image, poisson, fg_position_x_in_bg, fg_position_y_in_bg, fgsize)

                break
            elif tries >= 10:
                # rescale distractor if too large
                print("rescaling distractor")
                tries = 0
                fgsize_old = fgsize
                fgsize = ForegroundSize(int(round(fgsize.height * 0.9)),
                                        int(round(fgsize.width * 0.9)))
                if fgsize_old == fgsize:
                    # bricht aus while True aus, wenn Radarfalle sich über ganzes Bild erstreckt und daher kein distractor Objekt mehr hinpasst
                    break
                distractor_image = cv2.resize(distractor_image, (fgsize.width, fgsize.height),
                                              interpolation=cv2.INTER_AREA)

    def distractor_nonblured(self, distractor_image: np.ndarray, comb_image: CombinedImage, fg_position_x_in_bg: int,
                             fg_position_y_in_bg: int) -> None:
        mask: Image = Image.fromarray(distractor_image[:, :, 3])  # alpha Kanal

        # cv2 default is BGR --> conversion to RGB(A) for PIL
        # bg: Image = Image.fromarray(cv2.cvtColor(comb_image.binaries, cv2.COLOR_BGR2RGB))
        # fg: Image = Image.fromarray(cv2.cvtColor(distractor_image, cv2.COLOR_BGRA2RGBA))

        bg: Image = Image.fromarray(comb_image.binaries)
        fg: Image = Image.fromarray(distractor_image)

        bg.paste(fg, box=(fg_position_x_in_bg, fg_position_y_in_bg), mask=mask)
        comb_image.binaries = np.array(bg)

    def distractor_gauss(self, distractor_image: np.ndarray, comb_image: CombinedImage, fg_position_x_in_bg: int,
                         fg_position_y_in_bg: int) -> None:
        mask_gauss: np.ndarray = cv2.GaussianBlur(distractor_image[:, :, 3], (5, 5), 2)  # alpha Kanal
        mask: Image = Image.fromarray(mask_gauss)

        # cv2 default is BGR --> conversion to RGB(A) for PIL
        # bg: Image = Image.fromarray(cv2.cvtColor(comb_image.binaries, cv2.COLOR_BGR2RGB))
        # fg: Image = Image.fromarray(cv2.cvtColor(distractor_image, cv2.COLOR_BGRA2RGBA))

        bg: Image = Image.fromarray(comb_image.binaries)
        fg: Image = Image.fromarray(distractor_image)

        bg.paste(fg, box=(fg_position_x_in_bg, fg_position_y_in_bg), mask=mask)
        comb_image.binaries = np.array(bg)

    def distractor_poisson(self, distractor_image: np.ndarray, comb_image: CombinedImage, fg_position_x_in_bg: int,
                           fg_position_y_in_bg: int,
                           fgsize: ForegroundSize) -> None:
        # completely white mask keeps more details in foreground
        mask_poisson: np.ndarray = np.full((distractor_image.shape[0], distractor_image.shape[1]), 255).astype("uint8")
        # fg: np.ndarray = cv2.cvtColor(distractor_image[:, :, 0:3], cv2.COLOR_BGR2RGB)
        # bg: np.ndarray = cv2.cvtColor(comb_image.binaries, cv2.COLOR_BGR2RGB)
        fg: np.ndarray = distractor_image[:, :, 0:3]
        bg: np.ndarray = comb_image.binaries

        center: tuple = (fg_position_x_in_bg + fgsize.width // 2, fg_position_y_in_bg + fgsize.height // 2)
        new: np.ndarray = cv2.seamlessClone(fg, bg, mask_poisson, center, cv2.NORMAL_CLONE)
        comb_image.binaries = new

    def overlap(self, image: Union[Background, CombinedImage], position_x: int, position_y: int,
                size: ForegroundSize) -> float:
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

    def save_annotation_file(self, annotations, filename) -> None:
        with open(self.save_path + filename + ".txt", mode='w', newline='') as new_annotation_file:
            writer: csv.writer = csv.writer(new_annotation_file, delimiter=" ")
            for row in annotations:
                writer.writerow(row)


if __name__ == "__main__":
    source_path_fg = tk.filedialog.askdirectory(initialdir=INITDIR_FG,
                                                title="Choose source folder of foreground images")
    source_path_fg += "\\"
    source_path_bg = tk.filedialog.askdirectory(initialdir=INITDIR_BG,
                                                title="Choose source folder of background images")
    source_path_bg += "\\"
    save_path = tk.filedialog.askdirectory(initialdir=INITDIR_SAVE,
                                           title="Choose save folder for synthetically created images")
    save_path += "\\"

    if source_path_fg != "\\" and save_path != "\\" and source_path_bg != "\\":
        foreground_images: Queue
        distractor_objects: List[Foreground]
        foreground_images, distractor_objects = FGExecutor(source_path_fg).execute()
        print("fg and distrcator objects loaded...")
        background_images: Queue = BGExecutor(source_path_bg).execute()
        print("bg objects loaded...")
        Executor(foreground_images, background_images, save_path, distractor_objects).execute()

    # bg_test = Background("test", np.zeros((10, 10)), bounding_box_corners=[bounding_box(200, 210, 200, 210)])
    # size = ForegroundSize(20, 20)
    # ImageCombinator(Queue(), Queue(), "test", None).overlap(bg_test, 200, 200, size)
