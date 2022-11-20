import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import askyesno
import csv
import concurrent.futures

INITDIR = "F:\\RadarProjekt\\Training"
DEBUG = False


class Executor:
    def __init__(self, origin_path, save_path):
        self.origin_path = origin_path
        self.save_path = save_path
        self.images = []

    def start_augmentation(self):
        files = os.listdir(self.origin_path)
        for file in files:
            if file.endswith(".jpg"):
                self.images.append(file)

        with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
            for nr, image in enumerate(self.images):
                print(f"feeding {nr + 1} of {len(self.images)} files to thread pool")
                augment = Augmentation(image[:-4], self.origin_path, self.save_path)
                executor.submit(augment.augment, nr)


class Image:
    def __init__(self, name, binaries=None, annotations=None):
        self.binaries = binaries
        self.name = name
        self.annotations = [] if annotations is None else annotations


class Augmentation:
    def __init__(self, image_name, origin_path, save_path):
        self.image_name = image_name
        self.origin_path = origin_path
        self.save_path = save_path
        self.origin_image_size = 448
        self.final_image_size = 416
        self.brightness_increase_val = 30
        self.brightness_decrease_val = -30
        self.saturate_val = 30
        self.desaturate_val = -30

    def save_image(self, file, name, quality):
        pass
        # (1, 100) -> IMWRITE_JPEG_QUALITY 100 (highest)
        # cv2.imwrite(self.origin_path + image_name + NEW_IMAGE_SUFFIX, self.image_flipped, (1, 100))

    def augment(self, nr):
        print(f"{nr} started")
        templates = self.flip_images_horizontally()
        templates = self.translate_images(templates)
        self.increase_brightness(templates)
        self.decrease_brightness(templates)
        self.saturate(templates)
        self.desaturate(templates)
        print(f"{nr} completed")

    def flip_images_horizontally(self):
        bin_image = Image(self.image_name)
        bin_image.binaries = cv2.imread(self.origin_path + self.image_name + ".jpg", flags=-1)
        bin_image_flipped = Image(self.image_name + "_flipped")
        bin_image_flipped.binaries = np.fliplr(bin_image.binaries)
        images = [bin_image, bin_image_flipped]

        self.calc_annotation_flipped(bin_image, bin_image_flipped)
        return images

    def calc_annotation_flipped(self, bin_image, bin_image_flipped):
        annotation_file_path = self.origin_path + bin_image.name + ".txt"
        with open(annotation_file_path, mode='r') as old_annotation_file:
            reader = csv.reader(old_annotation_file, delimiter=" ")
            for row in reader:
                bin_image.annotations.append(row)
                shifted_x_value = str(1 - float(row[1]))
                new_row = [row[0], shifted_x_value, row[2], row[3], row[4]]
                bin_image_flipped.annotations.append(new_row)

    def translate_images(self, images):
        templates = []
        for image in images:
            translation1 = image.binaries[0:416, 0:416, :]  # row, column, depth
            cv2.imwrite(self.save_path + image.name + "_tr1.jpg", translation1, (1, 100))
            annotations = self.calc_annotations_translated(image, upper=True, left=True)
            self.save_txt(annotations, image.name + "_tr1.txt")
            templates.append(Image(image.name + "_tr1", translation1, annotations))

            translation2 = image.binaries[32:448, 0:416, :]
            cv2.imwrite(self.save_path + image.name + "_tr2.jpg", translation2, (1, 100))
            annotations = self.calc_annotations_translated(image, False, True)
            self.save_txt(annotations, image.name + "_tr2.txt")
            templates.append(Image(image.name + "_tr2", translation2, annotations))

            translation3 = image.binaries[0:416, 32:448, :]
            cv2.imwrite(self.save_path + image.name + "_tr3.jpg", translation3, (1, 100))
            annotations = self.calc_annotations_translated(image, True, False)
            self.save_txt(annotations, image.name + "_tr3.txt")
            templates.append(Image(image.name + "_tr3", translation3, annotations))

            translation4 = image.binaries[32:448, 32:448, :]
            cv2.imwrite(self.save_path + image.name + "_tr4.jpg", translation4, (1, 100))
            annotations = self.calc_annotations_translated(image, False, False)
            self.save_txt(annotations, image.name + "_tr4.txt")
            templates.append(Image(image.name + "_tr4", translation4, annotations))

            translation5 = image.binaries[16:432, 16:432, :]
            cv2.imwrite(self.save_path + image.name + "_tr5.jpg", translation5, (1, 100))
            annotations = self.calc_annotations_translated(image, center=True)
            self.save_txt(annotations, image.name + "_tr5.txt")
            templates.append(Image(image.name + "_tr5", translation5, annotations))

        return templates

    def calc_annotations_translated(self, image, upper=False, left=False, center=False):
        annotations = []
        for row in image.annotations:
            # <object-class> <x_center> <y_center> <width> <height>
            x_center = float(row[1]) * self.origin_image_size
            y_center = float(row[2]) * self.origin_image_size
            width = float(row[3]) * self.origin_image_size
            height = float(row[4]) * self.origin_image_size

            if center:  # (translation 5)
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
                if DEBUG:
                    print(
                        f"{image.name}: "
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
            annotations.append(new_row)
        return annotations

    def save_txt(self, annotations, filename):
        with open(self.save_path + filename, mode='w', newline='') as new_annotation_file:
            writer = csv.writer(new_annotation_file, delimiter=" ")
            for row in annotations:
                writer.writerow(row)

    def increase_brightness(self, templates):
        for image in templates:
            im = image.binaries
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.where(hsv[:, :, 2] > 255 - self.brightness_increase_val, 255,
                                    hsv[:, :, 2] + self.brightness_increase_val)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(self.save_path + image.name + "_bi.jpg", im, (1, 100))
            self.save_txt(image.annotations, image.name + "_bi.txt")

    def decrease_brightness(self, templates):
        for image in templates:
            im = image.binaries
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.where(hsv[:, :, 2] < 0 - self.brightness_decrease_val, 0,
                                    hsv[:, :, 2] + self.brightness_decrease_val)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(self.save_path + image.name + "_bd.jpg", im, (1, 100))
            self.save_txt(image.annotations, image.name + "_bd.txt")

    def saturate(self, templates):
        for image in templates:
            im = image.binaries
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.where(hsv[:, :, 1] > 255 - self.saturate_val, 255,
                                    hsv[:, :, 1] + self.saturate_val)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(self.save_path + image.name + "_sat.jpg", im, (1, 100))
            self.save_txt(image.annotations, image.name + "_sat.txt")

    def desaturate(self, templates):
        for image in templates:
            im = image.binaries
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.where(hsv[:, :, 1] < 0 - self.desaturate_val, 0,
                                    hsv[:, :, 1] + self.desaturate_val)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(self.save_path + image.name + "_desat.jpg", im, (1, 100))
            self.save_txt(image.annotations, image.name + "_desat.txt")


def delete_files(path):
    delstr = "_flipped"
    confirm = askyesno("Warning", f"Do you want to delete all files in path {path} containing \"{delstr}\"?")
    if confirm:
        files = os.listdir(path)
        for file in files:
            if delstr in file:
                os.remove(path + "\\" + file)


if __name__ == "__main__":
    original_path = tk.filedialog.askdirectory(initialdir=INITDIR, title="Choose source folder of images")
    original_path += "\\"
    save_path = tk.filedialog.askdirectory(initialdir=INITDIR,
                                           title="Choose save folder for augmented images")
    save_path += "\\"
    # delete_files(original_path)
    aug = Executor(original_path, save_path)
    aug.start_augmentation()
    print("augmentation finished")
