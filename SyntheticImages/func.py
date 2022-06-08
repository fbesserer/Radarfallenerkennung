# nicht dasselbe wie overlap method!
def intersection_over_union(self, image: Background, position_x: int, position_y: int, size: int) -> float:
    # calculate iou for all potential bboxes and the random_erase box
    max_overlap: float = 0
    erase_x1 = position_x
    erase_x2 = position_x + size * 2
    erase_y1 = position_y
    erase_y2 = position_y + size
    for bbox in image.bounding_box_corners:
        x1 = max(erase_x1, bbox.x1)
        x2 = min(erase_x2, bbox.x2)
        y1 = max(erase_y1, bbox.y1)
        y2 = min(erase_y2, bbox.y2)

        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        erase_area = (erase_x2 - erase_x1 + 1) * (erase_y2 - erase_y1 + 1)
        bbox_area = (bbox.x2 - bbox.x1 + 1) * (bbox.y2 - bbox.y1 + 1)
        iou = intersection_area / (erase_area + bbox_area - intersection_area)
        max_overlap = iou if iou > max_overlap else max_overlap

    return max_overlap
