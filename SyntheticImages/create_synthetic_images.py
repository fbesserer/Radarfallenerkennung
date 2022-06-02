import numpy as np
import cv2
from PIL import Image
from pb import *

# Poisson Blur
foreground = cv2.imread(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_26_original_6d3d361a-7abb-f040-4da3-555aea7d53bc_Kat2.png',
    cv2.IMREAD_UNCHANGED)

background_poisson = cv2.imread(
    "F:\\RadarProjekt\\Synthetische Bilder\\2020_12_24_original_0f3a831a-238b-a347-a6a6-0855a02654ff_Kat3.jpg")

mask_poisson = (np.full((358, 64), 255)).astype("uint8")
foreground = foreground[:, :, 0:3]
new_pic = cv2.seamlessClone(foreground, background_poisson, mask_poisson, (200, 200), cv2.MONOCHROME_TRANSFER)
cv2.imwrite("F:\\RadarProjekt\\Synthetische Bilder\\testPoisson.jpg", new_pic)

# blitzer unterschiedlich skalieren
new_size = (foreground.shape[1] // 2, foreground.shape[0] // 2)
resized = cv2.resize(foreground, new_size, interpolation=cv2.INTER_AREA)

# Gaussche Filtermaske
mask = foreground[:, :, 3]  # alpha Kanal
mask = cv2.GaussianBlur(mask, (5, 5), 2)
mask = Image.fromarray(mask)

background = Image.open(
    "F:\\RadarProjekt\\Synthetische Bilder\\2020_12_24_original_0f3a831a-238b-a347-a6a6-0855a02654ff_Kat3.jpg")

blitzermaske = Image.open(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_26_original_6d3d361a-7abb-f040-4da3-555aea7d53bc_Kat2.png')
try:
    background.paste(blitzermaske, box=(0, 0), mask=mask)
    background.save("F:\\RadarProjekt\\Synthetische Bilder\\testPILblured.jpg")
finally:
    background.close()
    blitzermaske.close()
