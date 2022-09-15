from typing import List, Tuple
import torch

from torch import nn, Tensor
from boxtargets import BoxTarget

INF = 100000000


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)
        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


# def clip_sigmoid(input):
#     out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)
#
#     return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out: Tensor, target: Tensor) -> Tensor:
        n_class: int = out.shape[1]
        class_ids: Tensor = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t: Tensor = target.unsqueeze(1)
        t_uniques = t.unique()
        p: Tensor = torch.sigmoid(out)

        gamma: float = self.gamma
        alpha: float = self.alpha

        term1: Tensor = (1 - p) ** gamma * torch.log(p)
        term2: Tensor = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss: Tensor = (
                -(t == class_ids).float() * alpha * term1
                - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )
        return loss.sum()


class FCOSLoss(nn.Module):
    def __init__(self, fpn_strides):
        super().__init__()

        self.sizes = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]  # entspricht mi in paper
        self.gamma = 2.0
        self.alpha = 0.25
        self.iou_loss_type = "giou"

        self.cls_loss = SigmoidFocalLoss(self.gamma, self.alpha)
        self.box_loss = IOULoss(self.iou_loss_type)
        self.center_loss = nn.BCEWithLogitsLoss()

        self.center_sample = True
        self.strides = fpn_strides
        self.radius = 1.5

    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
                top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )

        return torch.sqrt(centerness)

    def forward(self, locations, cls_pred, box_pred, center_pred, targets) -> Tuple[Tensor, ...]:
        batch: int = cls_pred[0].shape[0]
        n_class: int = cls_pred[0].shape[1]

        labels: List[Tensor, ...]
        box_targets: List[Tensor, ...]
        labels, box_targets = self.prepare_target(locations, targets)

        cls_flat: List[Tensor, ...] = []
        box_flat: List[Tensor, ...] = []
        center_flat: List[Tensor, ...] = []

        labels_flat: List[Tensor, ...] = []
        box_targets_flat: List[Tensor, ...] = []

        for i in range(len(labels)):
            cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))
            box_flat.append(box_pred[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred[i].permute(0, 2, 3, 1).reshape(-1))  # permute kreiert Kopie

            # label maps visualisieren
            # lbls = labels[i].reshape((cls_pred[i].shape[2], cls_pred[i].shape[3]))
            # lbls: ndarray = lbls.detach().cpu().numpy()
            # with open(f"csv{i}.csv", 'w', newline='') as file:
            #     writer = csv.writer(file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     for j in range(lbls.shape[0]):
            #         writer.writerow(lbls[j])

            labels_flat.append(labels[i].reshape(-1))  # reshape verändert
            box_targets_flat.append(box_targets[i].reshape(-1, 4))

        cls_flat: Tensor = torch.cat(cls_flat, 0)
        box_flat: Tensor = torch.cat(box_flat, 0)
        center_flat: Tensor = torch.cat(center_flat, 0)

        labels_flat: Tensor = torch.cat(labels_flat, 0)
        box_targets_flat: Tensor = torch.cat(box_targets_flat, 0)

        # pos_id: Tensor = torch.nonzero(labels_flat > 0).squeeze(1)
        pos_id: Tensor = torch.nonzero(labels_flat).squeeze(1)

        cls_loss: Tensor = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]

        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)

            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)

        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()

        return cls_loss, box_loss, center_loss

    def prepare_target(self, points: Tensor, targets: Tensor) -> (Tensor, Tensor):  # points = location
        ex_size_of_interest: List[Tensor, ...] = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])  # dtype und device with point_p_lvl
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
                # -1 = dimension 1 wird nicht verändert
            )

        ex_size_of_interest: Tensor = torch.cat(ex_size_of_interest, 0)
        n_point_per_level: List[int] = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first

    def compute_target_for_location(
            self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels: List = []
        box_targets: List = []
        xs, ys = locations[:, 0], locations[:, 1]

        for i in range(len(targets)):
            targets_per_img: BoxTarget = targets[i]
            assert targets_per_img.mode == 'xyxy'
            bboxes = targets_per_img.box
            labels_per_img = targets_per_img.fields['labels']
            area = targets_per_img.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img: Tensor = box_targets_per_img.max(dim=2)[0]

            is_cared_in_level = (
                                        max_box_targets_per_img >= sizes_of_interest[:, [0]]
                                ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(dim=1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]
            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        top = ys[:, None] - center_gt[..., 1]
        right = center_gt[..., 2] - xs[:, None]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes
