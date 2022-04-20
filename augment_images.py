import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog
import csv

ANNOTATION_FILES = []
NEW_IMAGE_SUFFIX = "_flipped.jpg"
NEW_ANNOTATION_SUFFIX = "_flipped.txt"
# INITDIR = "F:\\RadarProjekt\\Training"
INITDIR = "C:\\Users\\Fabian\\Documents\\MCSc\\Projekt\\Code\\DataAugmentation"


class Images:
    def __init__(self, path):
        self.path = path
        self.annotation_filename = ""
        self.original_image_size = 448
        self.final_image_size = 416

    def create_translated_images(self):
        files = os.listdir(self.path)
        for nr, file in enumerate(files):
            if file.endswith(".jpg"):
                im = cv2.imread(self.path + file, flags=-1)
                image_name = file[:-4]
                self.annotation_filename = self.path + image_name + ".txt"

                translation1 = im[0:416, 0:416, :]  # row, column, depth
                cv2.imwrite(self.path + image_name + "translated1.jpg", translation1, (1, 100))
                self.redraw_annotations("translated1.txt", upper=True, left=True)

                translation2 = im[32:448, 0:416, :]
                cv2.imwrite(self.path + image_name + "translated2.jpg", translation2, (1, 100))
                self.redraw_annotations("translated2.txt", False, True)

                translation3 = im[0:416, 32:448, :]
                cv2.imwrite(path + image_name + "translated3.jpg", translation3, (1, 100))
                self.redraw_annotations("translated3.txt", True, False)

                translation4 = im[32:448, 32:448, :]
                cv2.imwrite(path + image_name + "translated4.jpg", translation4, (1, 100))
                self.redraw_annotations("translated4.txt", False, False)

                translation5 = im[16:432, 16:432, :]
                cv2.imwrite(path + image_name + "translated5.jpg", translation5, (1, 100))
                self.redraw_annotations("translated5.txt", center=True)

    def redraw_annotations(self, suffix, upper=False, left=False, center=False):
        with open(self.annotation_filename[:-4] + suffix, mode='w', newline='') as new_annotation_file:
            with open(self.annotation_filename, mode='r') as old_annotation_file:
                reader = csv.reader(old_annotation_file, delimiter=" ")
                writer = csv.writer(new_annotation_file, delimiter=" ")
                for row in reader:
                    # <object-class> <x_center> <y_center> <width> <height>
                    x_center = float(row[1]) * self.original_image_size
                    y_center = float(row[2]) * self.original_image_size
                    width = float(row[3]) * self.original_image_size
                    height = float(row[4]) * self.original_image_size

                    if center:
                        x_center_new = (x_center - 16) / self.final_image_size
                        if x_center + 0.5 * width > self.final_image_size + 16:
                            width -= x_center + 0.5 * width - (self.final_image_size + 16)
                            if width < 0:
                                continue
                            x_center_new = (self.final_image_size - (0.5 * width)) / self.final_image_size
                        elif x_center - 0.5 * width < 16:
                            width -= 16 - (x_center - 0.5 * width)
                            if width < 0:
                                continue
                            x_center_new = (0.5 * width) / self.final_image_size

                        y_center_new = (y_center - 16) / self.final_image_size
                        if y_center + 0.5 * height > self.final_image_size + 16:
                            height -= y_center + 0.5 * height - (self.final_image_size + 16)
                            if height < 0:
                                continue
                            y_center_new = (self.final_image_size - (0.5 * height)) / self.final_image_size
                        elif y_center - 0.5 * height < 16:
                            height -= 16 - (y_center - 0.5 * height)
                            if height < 0:
                                continue
                            y_center_new = (0.5 * height) / self.final_image_size

                        print(
                            f"{self.annotation_filename[-20:]}: "
                            f"<{round(x_center_new * self.final_image_size - (width / 2), 1)}> "
                            f"<{round(x_center_new * self.final_image_size + (width / 2), 1)}> "
                            f"<{round(y_center_new * self.final_image_size - (height / 2), 1)}> "
                            f"<{round(y_center_new * self.final_image_size + (height / 2), 1)}>")
                    else:
                        if left:
                            x_center_new = x_center / self.final_image_size
                            if x_center + 0.5 * width > self.final_image_size:  # 410 + 10 = 420
                                width -= x_center + 0.5 * width - self.final_image_size  # 20 - (410 + 10 - 416) = 16
                                if width < 0:
                                    continue
                                x_center_new = (self.final_image_size - (0.5 * width)) / self.final_image_size
                        else:
                            x_center_new = (x_center - 32) / self.final_image_size
                            if x_center - 0.5 * width < 32:
                                width -= 32 - (x_center - 0.5 * width)  # 20 - ( 32 - (40 - 10))
                                if width < 0:
                                    continue
                                x_center_new = (0.5 * width) / self.final_image_size

                        if upper:
                            y_center_new = y_center / self.final_image_size
                            if y_center + 0.5 * height > self.final_image_size:
                                height -= y_center + 0.5 * height - self.final_image_size
                                if height < 0:
                                    continue
                                y_center_new = (self.final_image_size - (0.5 * height)) / self.final_image_size
                        else:
                            y_center_new = (y_center - 32) / self.final_image_size
                            if y_center - 0.5 * height < 32:
                                height -= 32 - (y_center - 0.5 * height)
                                if height < 0:
                                    continue
                                y_center_new = (0.5 * height) / self.final_image_size

                    width = min(width / self.final_image_size, 1)
                    height = min(height / self.final_image_size, 1)

                    new_row = [row[0], "{:.6f}".format(x_center_new), "{:.6f}".format(y_center_new),
                               "{:.6f}".format(width), "{:.6f}".format(height)]
                    writer.writerow(new_row)


# Todo: funktion die bilder öffnet, sämtl augmentierungs funktion machen dann nur die jeweilige augmentierung
def flip_images_horizontally(path, files):
    for nr, file in enumerate(files):
        print(f"looping through {nr + 1} of {len(files)} files")
        if file.endswith(".jpg"):
            im = cv2.imread(path + file, flags=-1)
            im = np.fliplr(im)
            image_name = file[:-4]
            ANNOTATION_FILES.append(path + image_name + ".txt")
            # (1, 100) -> IMWRITE_JPEG_QUALITY 100 (highest)
            cv2.imwrite(path + image_name + NEW_IMAGE_SUFFIX, im, (1, 100))


def create_annotation_files():
    for nr, file in enumerate(ANNOTATION_FILES):
        print(f"looping through {nr + 1} of {len(ANNOTATION_FILES)} annotation files")
        with open(file[:-4] + NEW_ANNOTATION_SUFFIX, mode='w', newline='') as new_annotation_file:
            with open(file, mode='r') as old_annotation_file:
                reader = csv.reader(old_annotation_file, delimiter=" ")
                writer = csv.writer(new_annotation_file, delimiter=" ")
                for row in reader:
                    shifted_x_value = str(1 - float(row[1]))
                    new_row = [row[0], shifted_x_value, row[2], row[3], row[4]]
                    writer.writerow(new_row)


def delete_files(path):
    files = os.listdir(path)
    for file in files:
        if "translated" in file:
            os.remove(path + "\\" + file)


if __name__ == "__main__":
    path = tk.filedialog.askdirectory(initialdir=INITDIR, title="Ordner mit Bildern auswählen")
    path += "\\"
    delete_files(path)
    # flip_images_horizontally(path, os.listdir(path))
    # create_annotation_files()
    # images = Images(path)
    # images.create_translated_images()
