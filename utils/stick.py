"""
Functions used to stick a patch on an X-ray image.
"""


def parse_gtbox(targets):
    """
    Parse the information of ground truth box.
    Only support one ground truth box per image.
    Return: gtbox (x, y, w, h) in 1x1 scale.
    """
    gt_boxes = []
    for s in range(len(targets)):
        # ground_truth: (tu, tv) - (tu+tw, tv+th)
        # patch: (u, v) - (u+W, v+H)
        elm = targets[s]
        tu = elm[0][1]
        tv = elm[0][0]
        tw = elm[0][3]
        th = elm[0][2]
        tw = tw - tu
        th = th - tv
        
        gt_boxes.append([tu, tv, tw, th])

    return gt_boxes


def cal_stick_place(gt_boxes, W, H, overlap, direction="nw"):
    """
    Calculate the specific stick place.
    Return: specific place through direction and overlap
    [(x, y)] for left-top coord in 300x300 scale
    """
    places = []
    for box in gt_boxes:
        if direction == "center":
            u = 150 - W // 2
            v = 150 - H // 2
            places.append([int(u), int(v)])
            continue
        
        if direction == "top":
            u = 0
            v = 0
            places.append([int(u), int(v)])
            continue
            
        tw = int(box[2] * 300)
        th = int(box[3] * 300)
        tu = int(box[0] * 300)
        tv = int(box[1] * 300)
        
        if direction == "cover":
            u = tu + tw / 2 - W / 2
            v = tv + th / 2 - H / 2
            places.append([int(u), int(v)])
            continue
        
        
        if "w" in direction:
            u = tu + overlap * tw - W
        elif "e" in direction:
            u = tu + (1-overlap) * tw
        else:
            u = tu + tw / 2 - W / 2
            
        if "n" in direction:
            v = tv + overlap * th - H
        elif "s" in direction:
            v = tv + (1-overlap) * th
        else:
            v = tv + th / 2 - H / 2
            
        places.append([int(u), int(v)])

    return places


def cal_stick_place_rcnn(gt_boxes, W, H, overlap, direction="nw"):
    """
    Calculate the specific stick place.
    Input gt_boxes: [N, 4] -> (ymin, xmin, ymax, xmax)
    Return: specific place through direction and overlap
    [(x, y)] for left-top coord in resized scale (short size 300)
    """
    places = []
    for box in gt_boxes:
        tu = int(box[0][0])
        tv = int(box[0][1])
        tw = int(box[0][2]) - tu
        th = int(box[0][3]) - tv
        
        if "w" in direction:
            u = tu + overlap * tw - W
        elif "e" in direction:
            u = tu + (1-overlap) * tw
        else:
            u = tu + tw / 2 - W / 2
            
        if "n" in direction:
            v = tv + overlap * th - H
        elif "s" in direction:
            v = tv + (1-overlap) * th
        else:
            v = tv + th / 2 - H / 2
            
        places.append([int(u), int(v)])

    return places


def get_stick_area(gt_boxes, w, h):
    """
    Calculate the recommended stick area.
    Return: discrete pastable area coordinate [N * (x, y)] in 300x300 scale
    """
    areas = []
    for box in gt_boxes:
        tw = int((box[2] / 2) * 300) + w
        th = int((box[3] / 2) * 300) + h
        tu = int((box[0] + box[2] / 4) * 300) - w
        tv = int((box[1] + box[3] / 4) * 300) - h

        tx = tu + tw
        ty = tv + th

        area = []
        area.extend([(i, tv) for i in range(tu, tx, 1)])
        area.extend([(tx, i) for i in range(tv, ty, 1)])
        area.extend([(i, ty) for i in range(tx, tu, -1)])
        area.extend([(tu, i) for i in range(ty, tv, -1)])
        
        areas.append(area)

    return areas
        

def get_stick_area_rcnn(gt_boxes, w, h):
    """
    Calculate the recommended stick area.
    Return: discrete pastable area coordinate [N * (x, y)] in short size 300 scale
    """
    areas = []
    for box in gt_boxes:
        tw = int((box[0][2] - box[0][0]) / 2) + w
        th = int((box[0][3] - box[0][1]) / 2) + h
        tu = int((box[0][0] + (box[0][2] - box[0][0]) / 4)) - w
        tv = int((box[0][1] + (box[0][3] - box[0][1]) / 4)) - h

        tx = tu + tw
        ty = tv + th

        area = []
        area.extend([(i, tv) for i in range(tu, tx, 1)])
        area.extend([(tx, i) for i in range(tv, ty, 1)])
        area.extend([(i, ty) for i in range(tx, tu, -1)])
        area.extend([(tu, i) for i in range(ty, tv, -1)])
        
        areas.append(area)

    return areas