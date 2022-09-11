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
