import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm
from boxtargets import BoxTarget

IMG_FORMATS = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


class ImagesAndLabels(Dataset):
    def __init__(self, path, cache_images=False):
        path = str(Path(path))  # os-agnostic
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in IMG_FORMATS]

        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]
        n = len(self.img_files)

        # Cache labels
        self.imgs = [None] * n
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n  # 1 label wert + x,y,w,h
        pbar = tqdm(self.label_files, desc='Caching labels')
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        for i, file in enumerate(pbar):
            try:
                with open(file, 'r') as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            except:
                nm += 1  # file missing
                continue

            if l.shape[0]:  # check ob alle annotations korrekt (label, x,y,w,h)
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # duplicate rows
                self.labels[i] = l
                nf += 1  # file found

            else:
                ne += 1  # file empty

            pbar.desc = 'Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                nf, nm, ne, nd, n)
        assert nf > 0, 'No labels found in %s' % (os.path.dirname(file) + os.sep)

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0 = [None] * n
            # self.img_hw = [None] * n
            for i in pbar:
                self.imgs[i], self.img_hw0[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        # Load image
        img, (h0, w0) = load_image(self, idx)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)  # Speicheroptimierung
        img = torch.tensor(img, dtype=torch.float32)

        # Load labels
        labels = []
        x = self.labels[idx]
        if x.size > 0:
            # von Normalized xywh to pixel xyxy format
            labels = x.copy()
            labels[:, 1] = w0 * (x[:, 1] - x[:, 3] / 2)  # warum 1-4? weil 0 == label
            labels[:, 2] = h0 * (x[:, 2] - x[:, 4] / 2)
            labels[:, 3] = w0 * (x[:, 1] + x[:, 3] / 2)
            labels[:, 4] = h0 * (x[:, 2] + x[:, 4] / 2)

        # BoxList Objekt erstellen, ursprüngliche Konvertierung war von nicht normalisierten xywh auf xyxy
        boxes: List = [box[1:] for box in labels]  # nur die xyxy Werte übernehmen
        boxes: Tensor = torch.as_tensor(boxes).reshape(-1, 4)
        target: BoxTarget = BoxTarget(boxes, (img.shape[1], img.shape[2]), mode='xyxy')

        classes: List = [label[0] for label in labels]
        classes: Tensor = torch.tensor(classes)
        target.fields['labels'] = classes

        target.clip(remove_empty=True)  # über Bildmaße hinausragende Boxes beschneiden

        return img, target, idx


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        # r = self.img_size / max(h0, w0)  # resize image to img_size
        # if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
        #     interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0)  # , img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index]  # , self.img_hw[index]  # img, hw_original, hw_resized


class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)


def image_list(tensors):
    shape = (len(tensors), 3, 416, 416)
    batch = tensors[0].new(*shape).zero_()
    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    sizes = [img.shape[-2:] for img in tensors]
    return ImageList(batch, sizes)


def collate_fn():
    def collate_data(batch):
        # formatiert die Daten nach __get_item__ und gibt sie an die train() for loop zurück
        batch = list(zip(*batch))
        imgs = image_list(batch[0])
        targets = batch[1]
        ids = batch[2]

        return imgs, targets, ids

    return collate_data

# def transform_to_dict(target, id) -> dict:
#     # transforms the BoxTarget Object into normal dict in order to be able to use pytorch evaluate()
#     d = {}
#     target = target[0]
#     d['boxes'] = target.box
#     d['labels'] = target.fields['labels']
#     d['image_id'] = id
#     d['area'] = target.area()
#     d['iscrowd'] = False
#     return d
#
#
# def collate_fn_eval():  # überprüfen ob es auch mit batchsize > 1 geht
#     def collate_data(batch):
#         batch = list(zip(*batch))
#         imgs = image_list(batch[0]).tensors
#         targets = batch[1]
#         ids = batch[2]
#         targets = transform_to_dict(targets, ids)
#
#         return imgs, targets
#
#     return collate_data

# # war ursprünglich für pytorch coco_eval etc Skripte gedacht
# class ImagesAndLabelsValidationSet(Dataset):
#     def __init__(self, path):
#         self.path = str(Path(path))
#         with open(path, 'r') as f:
#             self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
#                               if os.path.splitext(x)[-1].lower() in IMG_FORMATS]
#
#         self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
#                             for x in self.img_files]
#         n = len(self.img_files)
#
#         # Cache labels
#         self.imgs = [None] * n
#         self.labels = [np.zeros((0, 5), dtype=np.float32)] * n  # 1 label wert + x,y,w,h
#         pbar = tqdm(self.label_files, desc='Caching labels')
#         nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
#         for i, file in enumerate(pbar):
#             try:
#                 with open(file, 'r') as f:
#                     l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
#             except:
#                 nm += 1  # file missing
#                 continue
#
#             if l.shape[0]:  # check ob alle annotations korrekt (label, x,y,w,h)
#                 assert l.shape[1] == 5, '> 5 label columns: %s' % file
#                 assert (l >= 0).all(), 'negative labels: %s' % file
#                 assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
#                 if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
#                     nd += 1  # duplicate rows
#                 self.labels[i] = l
#                 nf += 1  # file found
#
#             else:
#                 ne += 1  # file empty
#
#             pbar.desc = 'Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
#                 nf, nm, ne, nd, n)
#         assert nf > 0, 'No labels found in %s' % (os.path.dirname(file) + os.sep)
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         # self.imgs = list(sorted(os.listdir(os.path.join(path, "PNGImages"))))
#         # self.masks = list(sorted(os.listdir(os.path.join(path, "PedMasks"))))
#
#     def __getitem__(self, idx):
#         # Load image
#         img, (h0, w0) = load_image(self, idx)
#         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         img = np.ascontiguousarray(img)  # Speicheroptimierung
#         img = torch.tensor(img, dtype=torch.float32)
#
#         # Load labels
#         labels = []
#         x = self.labels[idx]
#         if x.size > 0:
#             # von Normalized xywh to pixel xyxy format
#             labels = x.copy()
#             labels[:, 1] = w0 * (x[:, 1] - x[:, 3] / 2)  # warum 1-4? weil 0 == label
#             labels[:, 2] = h0 * (x[:, 2] - x[:, 4] / 2)
#             labels[:, 3] = w0 * (x[:, 1] + x[:, 3] / 2)
#             labels[:, 4] = h0 * (x[:, 2] + x[:, 4] / 2)
#         # BoxList Objekt erstellen, ursprüngliche Konvertierung war von nicht normalisierten xywh auf xyxy
#         boxes: List = [box[1:] for box in labels]  # nur die xyxy Werte übernehmen
#         boxes: Tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         classes: List = [label[0] for label in labels]
#         classes: Tensor = torch.tensor(classes, dtype=torch.int64)
#
#         # convert everything into a torch.Tensor
#         # boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # there is only one class -> daher nur torch.ones
#         # labels = torch.ones((num_objs,), dtype=torch.int64)
#         # masks = torch.as_tensor(masks, dtype=torch.uint8)
#
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
#
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = classes
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd
#
#         return img, target
#
#     def __len__(self):
#         return len(self.img_files)
