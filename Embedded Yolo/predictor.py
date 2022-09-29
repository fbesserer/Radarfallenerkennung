import argparse
import os
import time

import cv2
import torch
from network import EmbeddedYolo
from dataset import ImageList


class Inference(object):
    CATEGORIES = [
        "kat1_AnhaengerMobil",
        "kat2_Sauelenblitzer",
        "kat3_KastenMobil",
        "kat5_Starenkasten"
    ]

    def __init__(
            self,
            opt,
            confidence_thresholds_for_classes,
            show_mask_heatmaps=False,
            masks_per_dim=2,
            min_image_size=416,
    ):
        self.opt = opt
        self.model = EmbeddedYolo(opt)
        self.model.load_state_dict(torch.load(opt.weights)['model'])
        self.model.eval()
        self.device = torch.device('cuda:0')
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = opt.output  # todo ordner löschen wenn bereits besteht
        # checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        # _ = checkpointer.load(cfg.MODEL.WEIGHT)
        #
        # self.transforms = self.build_transform()

        # mask_threshold = -1 if show_mask_heatmaps else 0.5
        # self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_thresholds_for_classes = torch.tensor((confidence_thresholds_for_classes,) * 4)
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        # todo non maxium suppresion umsetzen

        result = image.copy()
        # if self.show_mask_heatmaps:
        #     return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        # if self.cfg.MODEL.MASK_ON:
        #     result = self.overlay_mask(result, top_predictions)
        # if self.cfg.MODEL.KEYPOINT_ON:
        #     result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        # image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        # image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        # image_list = image_list.to(self.device)
        image_list = ImageList(torch.tensor(original_image), ((416, 416),))
        image_list.tensors = image_list.tensors.permute(2, 0, 1).float()
        image_list.tensors = image_list.tensors[None, :]
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list.tensors, image_list.sizes, detection=True)
        # predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0][0]
        prediction.box = prediction.box.to(self.cpu_device)
        prediction.fields['scores'] = prediction.fields['scores'].to(self.cpu_device)
        prediction.fields['labels'] = prediction.fields['labels'].to(self.cpu_device)

        # reshape prediction (a BoxList) into the original image size
        # height, width = original_image.shape[:-1]
        # prediction = prediction.resize((width, height))

        # if prediction.has_field("mask"):
        #     # if we have masks, paste the masks in the right position
        #     # in the image, as defined by the bounding boxes
        #     masks = prediction.get_field("mask")
        #     # always single image is passed at a time
        #     masks = self.masker([masks], [prediction])[0]
        #     prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.fields["scores"]
        labels = predictions.fields["labels"]
        thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        # scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.fields["labels"]
        boxes = predictions.box

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.fields["scores"].tolist()
        labels = predictions.fields["labels"].tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.box

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='checkpoint/epoch-9.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--data', type=str, default='data/radar.data', help='*.data path')

    opt = parser.parse_args()
    print(opt)
    opt.half = False

    opt.n_class = 5
    opt.top_n = 1000
    opt.post_top_n = 100
    opt.min_size = 0

    im_names = os.listdir(opt.source)

    # prepare object that handles inference plus adds predictions on top of image
    inf_class = Inference(
        opt,
        confidence_thresholds_for_classes=opt.conf_threshold,
    )

    for im_name in im_names:
        img = cv2.imread(os.path.join(opt.source, im_name))
        if img is None:
            continue
        start_time = time.time()
        composite = inf_class.run_on_opencv_image(img)
        print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
        # todo save all
        cv2.imwrite(os.path.join(opt.output, im_name), composite)

        # cv2.imshow(im_name, composite)
    print("Press any keys to exit ...")
    # cv2.waitKey()
    # cv2.destroyAllWindows()
