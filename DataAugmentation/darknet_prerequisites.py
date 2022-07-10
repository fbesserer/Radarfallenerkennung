import tkinter as tk
from tkinter import filedialog
import os
import csv

INITDIR = "F:\\RadarProjekt\\Training"
TXT_NAME= "train.txt"
# TXT_NAME = "test.txt"
# TXT_NAME = "valid.txt"


def create_txt(path):
    rows = []
    files = os.listdir(path)
    for file in files:
        if ".jpg" in file:
            rows.append(["data/obj/" + file])
    with open(path + TXT_NAME, mode='w', newline='') as txt:
        writer = csv.writer(txt, delimiter=" ")
        writer.writerows(rows)
    print(f"{TXT_NAME} unter {path} erstellt.")


if __name__ == "__main__":
    path = tk.filedialog.askdirectory(initialdir=INITDIR, title="Ordner mit Bildern auswählen")
    path += "\\"

    create_txt(path)
