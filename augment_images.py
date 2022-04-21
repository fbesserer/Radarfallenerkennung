import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import askyesno
import csv

NEW_IMAGE_SUFFIX = "_flipped.jpg"
# NEW_ANNOTATION_SUFFIX = "_flipped.txt"
# INITDIR = "F:\\RadarProjekt\\Training"
INITDIR = "C:\\Users\\Fabian\\Documents\\MCSc\\Projekt\\Code\\DataAugmentation"


class Images:
    def __init__(self, origin_path, save_path):
        self.origin_path = origin_path
        self.save_path = save_path
        self.images = []

    def start_augmentation(self):
        files = os.listdir(self.origin_path)
        for file in files:
            if file.endswith(".jpg"):
                self.images.append(file)

        for nr, image in enumerate(self.images):
            print(f"looping through {nr + 1} of {len(self.images)} files")
            augment = Augmentation(image, self.origin_path, self.save_path)
            augment.flip_images_horizontally()


class Augmentation:
    def __init__(self, image_name, origin_path, save_path):
        self.image_name = image_name
        self.origin_path = origin_path
        self.save_path = save_path
        self.image_bin = None
        self.image_flipped_bin = None
        self.annotation_file_path = ""
        self.annotation_file_path_flipped = ""
        self.newly_created_annot_files = []
        self.origin_image_size = 448
        self.final_image_size = 416

    def save_image(self):
        pass
        # (1, 100) -> IMWRITE_JPEG_QUALITY 100 (highest)
        # cv2.imwrite(self.origin_path + image_name + NEW_IMAGE_SUFFIX, self.image_flipped, (1, 100))

    def flip_images_horizontally(self):
        self.image_bin = cv2.imread(self.origin_path + self.image_name, flags=-1)
        self.image_flipped_bin = np.fliplr(self.image_bin)

        self.annotation_file_path = self.origin_path + self.image_name[:-4] + ".txt"
        self.annotation_file_path_flipped = self.origin_path + self.annotation_file_path[:-4] + "_flipped.txt"
        self.create_annotation_file()
        self.translate_images()

    def create_annotation_file(self):
        with open(self.annotation_file_path_flipped, mode='w', newline='') as new_annotation_file:
            with open(self.annotation_file_path, mode='r') as old_annotation_file:
                reader = csv.reader(old_annotation_file, delimiter=" ")
                writer = csv.writer(new_annotation_file, delimiter=" ")
                for row in reader:
                    shifted_x_value = str(1 - float(row[1]))
                    new_row = [row[0], shifted_x_value, row[2], row[3], row[4]]
                    writer.writerow(new_row)
        self.newly_created_annot_files.append(self.annotation_file_path_flipped)  # collect for later deletion

    def translate_images(self):
        # for nr, file in enumerate(self.files):
        for _ in range(2):
            # if file.endswith(".jpg"):
            # im = cv2.imread(self.origin_path + file, flags=-1)
            # image_name = file[:-4]
            # self.annotation_file_path = image_name + ".txt"

            translation1 = im[0:416, 0:416, :]  # row, column, depth
            cv2.imwrite(self.save_path + image_name + "translated1.jpg", translation1, (1, 100))
            self.redraw_annotations("translated1.txt", upper=True, left=True)

            translation2 = im[32:448, 0:416, :]
            cv2.imwrite(self.save_path + image_name + "translated2.jpg", translation2, (1, 100))
            self.redraw_annotations("translated2.txt", False, True)

            translation3 = im[0:416, 32:448, :]
            cv2.imwrite(self.save_path + image_name + "translated3.jpg", translation3, (1, 100))
            self.redraw_annotations("translated3.txt", True, False)

            translation4 = im[32:448, 32:448, :]
            cv2.imwrite(self.save_path + image_name + "translated4.jpg", translation4, (1, 100))
            self.redraw_annotations("translated4.txt", False, False)

            translation5 = im[16:432, 16:432, :]
            cv2.imwrite(self.save_path + image_name + "translated5.jpg", translation5, (1, 100))
            self.redraw_annotations("translated5.txt", center=True)

    def redraw_annotations(self, suffix, upper=False, left=False, center=False):
        with open(self.save_path + self.annotation_file_path[:-4] + suffix, mode='w',
                  newline='') as new_annotation_file:
            with open(self.origin_path + self.annotation_file_path, mode='r') as old_annotation_file:
                reader = csv.reader(old_annotation_file, delimiter=" ")
                writer = csv.writer(new_annotation_file, delimiter=" ")
                for row in reader:
                    # <object-class> <x_center> <y_center> <width> <height>
                    x_center = float(row[1]) * self.origin_image_size
                    y_center = float(row[2]) * self.origin_image_size
                    width = float(row[3]) * self.origin_image_size
                    height = float(row[4]) * self.origin_image_size

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
                            f"{self.annotation_file_path}: "
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


def delete_files(path):
    confirm = askyesno("Warning", f"Do you want to delete all files in path {path} with \"translated\" in their name?")
    if confirm:
        files = os.listdir(path)
        for file in files:
            if "translated" in file:
                os.remove(path + "\\" + file)


if __name__ == "__main__":
    original_path = tk.filedialog.askdirectory(initialdir=INITDIR, title="Ordner mit Bildern auswählen")
    original_path += "\\"
    save_path = tk.filedialog.askdirectory(initialdir=INITDIR,
                                           title="Ordner auswählen, an dem augmentierte Bilder gespeichert werden sollen")
    save_path += "\\"
    # delete_files(original_path)
    # flip_images_horizontally(original_path, os.listdir(original_path))
    # create_annotation_file()
    aug = Images(original_path, save_path)
    aug.start_augmentation()
