import numpy as np
import cv2

blitzermaske = cv2.imread(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_26_original_6d3d361a-7abb-f040-4da3-555aea7d53bc_Kat2.png',
    cv2.IMREAD_UNCHANGED)
background = cv2.imread(
    'F:\\RadarProjekt\\Synthetische Bilder\\2020_12_24_original_0f3a831a-238b-a347-a6a6-0855a02654ff_Kat3.jpg')
background = np.dstack((background, np.zeros((448, 448))))
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

# Maske für cv2.add() = umgedrehter Transparenzkanal
mask = (blitzermaske_correct_size[:, :, 3])
mask = (np.where(mask > 0, 0, 255)).astype("uint8")
# loch aus background ausschneiden
result = cv2.add(blitzermaske_correct_size, background, mask=mask)
# blitzer einfügen
result = cv2.add(blitzermaske_correct_size, result)

# todo: blending operationen ausführen

cv2.imwrite("F:\\RadarProjekt\\Synthetische Bilder\\test.jpg", result)
