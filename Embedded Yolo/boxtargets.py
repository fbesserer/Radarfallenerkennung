import torch
from torchvision import ops


class BoxTarget:
    def __init__(self, boxes, image_size, mode='xyxy'):
        device = boxes.device if hasattr(boxes, 'device') else 'cpu'
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)

        self.box = boxes
        self.size = image_size
        self.mode = mode

        self.fields = {}

    def __len__(self):
        return self.box.shape[0]

    def __getitem__(self, index):
        box = BoxTarget(self.box[index], self.size, self.mode)

        for k, v in self.fields.items():
            box.fields[k] = v[index]

        return box

    def clip(self, remove_empty=True):
        remove = 1

        max_width = self.size[0] - remove
        max_height = self.size[1] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_width)
        self.box[:, 3].clamp_(min=0, max=max_height)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])

            return self[keep]

        else:
            return self

    def to(self, device):
        box = BoxTarget(self.box.to(device), self.size, self.mode)

        for k, v in self.fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)

            box.fields[k] = v

        return box

    def area(self):
        box = self.box

        if self.mode == 'xyxy':
            remove = 1

            area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)

        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]

        return area

    def convert(self, mode):
        if mode == self.mode:
            return self

        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if mode == 'xyxy':
            box = torch.cat([x_min, y_min, x_max, y_max], -1)
            box = BoxTarget(box, self.size, mode=mode)

        elif mode == 'xywh':
            remove = 1
            box = torch.cat(
                [x_min, y_min, x_max - x_min + remove, y_max - y_min + remove], -1
            )
            box = BoxTarget(box, self.size, mode=mode)

        box.copy_field(self)

        return box

    def split_to_xyxy(self):
        if self.mode == 'xyxy':
            x_min, y_min, x_max, y_max = self.box.split(1, dim=-1)

            return x_min, y_min, x_max, y_max

        elif self.mode == 'xywh':
            remove = 1
            x_min, y_min, w, h = self.box.split(1, dim=-1)

            return (
                x_min,
                y_min,
                x_min + (w - remove).clamp(min=0),
                y_min + (h - remove).clamp(min=0),
            )

    def copy_field(self, box):
        for k, v in box.fields.items():
            self.fields[k] = v


def remove_small_box(boxlist, min_size):
    box = boxlist.convert('xywh').box
    _, _, w, h = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]


def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    box = boxlist.box
    keep = ops.nms(box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep]

    return boxlist.convert(mode)


def cat_boxlist(boxlists):
    size = boxlists[0].size
    mode = boxlists[0].mode
    field_keys = boxlists[0].fields.keys()

    box_cat = torch.cat([boxlist.box for boxlist in boxlists], 0)
    new_boxlist = BoxTarget(box_cat, size, mode)

    for field in field_keys:
        data = torch.cat([boxlist.fields[field] for boxlist in boxlists], 0)
        new_boxlist.fields[field] = data

    return new_boxlist
