import numpy as np
import cv2
from PIL import Image

blitzermaske = cv2.imread(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_26_original_6d3d361a-7abb-f040-4da3-555aea7d53bc_Kat2.png',
    cv2.IMREAD_UNCHANGED)

# blitzer unterschiedlich skalieren
new_size = (blitzermaske.shape[1] // 2, blitzermaske.shape[0] // 2)
resized = cv2.resize(blitzermaske, new_size, interpolation=cv2.INTER_AREA)

# background = cv2.imread(
#     'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_24_original_0f3a831a-238b-a347-a6a6-0855a02654ff_Kat3.jpg')
# background = np.dstack((background, np.zeros((448, 448))))

# transparency maske erstellen 448x448
blitzermaske_correct_size = np.zeros((448, 448, 4))

# Blitzermaske einfügen über slicing, RGB Kanäle abhängig vom Transparenzkanal (4. Kanal) auf 0 setzen
blitzermaske_correct_size[10:10 + blitzermaske.shape[0], 200:200 + blitzermaske.shape[1], :] = blitzermaske
blitzermaske_correct_size[:, :, 0] = np.where(blitzermaske_correct_size[:, :, 3] > 0,
                                              blitzermaske_correct_size[:, :, 0], 0)
blitzermaske_correct_size[:, :, 1] = np.where(blitzermaske_correct_size[:, :, 3] > 0,
                                              blitzermaske_correct_size[:, :, 1], 0)
blitzermaske_correct_size[:, :, 2] = np.where(blitzermaske_correct_size[:, :, 3] > 0,
                                              blitzermaske_correct_size[:, :, 2], 0)

mask = (blitzermaske_correct_size[:, :, 3]).astype("uint8")
# Maske für cv2.add() = umgedrehter Transparenzkanal
# mask = (np.where(mask > 0, 0, 255)).astype("uint8")

mask = cv2.GaussianBlur(mask, (5, 5), 2)
# mask = mask[:, :, np.newaxis]
# mask = np.repeat(mask, 3, axis=2)
mask = Image.fromarray(mask)

with Image.open(
        "F:\\RadarProjekt\\Synthetische Bilder\\2020_12_24_original_0f3a831a-238b-a347-a6a6-0855a02654ff_Kat3.jpg") as im:
    blitzermaske_correct_size = (blitzermaske_correct_size[:, :, 0:3]).astype("uint8")
    blitzermaske_correct_size = Image.fromarray(blitzermaske_correct_size)
    im.paste(blitzermaske_correct_size, box=(0, 0), mask=mask)
    im.save("F:\\RadarProjekt\\Synthetische Bilder\\testPILblured.png")

# # loch aus background ausschneiden
# result = cv2.add(blitzermaske_correct_size, background, mask=mask)
# # blitzer einfügen
# result = cv2.add(blitzermaske_correct_size, result)

cv2.imwrite("F:\\RadarProjekt\\Synthetische Bilder\\test.jpg", result)
