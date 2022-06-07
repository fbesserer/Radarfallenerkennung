import numpy as np
import cv2
from PIL import Image

# Poisson Blur
foreground = cv2.imread(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_26_original_6d3d361a-7abb-f040-4da3-555aea7d53bc_Kat2.png',
    cv2.IMREAD_UNCHANGED)

background_poisson = cv2.imread(
    "F:\\RadarProjekt\\Synthetische Bilder\\2020_12_24_original_0f3a831a-238b-a347-a6a6-0855a02654ff_Kat3.jpg")

mask_poisson = (np.full((foreground.shape[0], foreground.shape[1]), 255)).astype(
    "uint8")  # komplett weiße Maske erhält mehr Details
# mask_poisson = foreground[:, :, 3].astype("uint8")
foreground = foreground[:, :, 0:3]
new_pic = cv2.seamlessClone(foreground, background_poisson, mask_poisson, (200, 200),
                            cv2.NORMAL_CLONE)  # (200,200) == center
cv2.imwrite("F:\\RadarProjekt\\Synthetische Bilder\\testPoisson.jpg", new_pic)

# Gaussche Filtermaske
foreground = cv2.imread(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_26_original_6d3d361a-7abb-f040-4da3-555aea7d53bc_Kat2.png',
    cv2.IMREAD_UNCHANGED)
mask = foreground[:, :, 3]  # alpha Kanal
mask_gauss = cv2.GaussianBlur(mask, (5, 5), 2)
# mask = Image.fromarray(mask)
mask_gauss = Image.fromarray(mask_gauss)

background = Image.open(
    "F:\\RadarProjekt\\Synthetische Bilder\\2020_12_24_original_0f3a831a-238b-a347-a6a6-0855a02654ff_Kat3.jpg")
# background_noblur = background.copy()
blitzermaske = Image.open(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_26_original_6d3d361a-7abb-f040-4da3-555aea7d53bc_Kat2.png')

background.paste(blitzermaske, box=(0, 0), mask=mask_gauss)
background.save("F:\\RadarProjekt\\Synthetische Bilder\\testPILblured.jpg")
# background_noblur.paste(blitzermaske, box=(0, 0), mask=mask)
# background_noblur.save("F:\\RadarProjekt\\Synthetische Bilder\\testPILnoblur.jpg")
background.close()
blitzermaske.close()
