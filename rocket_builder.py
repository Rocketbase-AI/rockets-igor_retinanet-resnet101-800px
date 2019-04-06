import os
from .architecture import resnet50
import torch
from torchvision import transforms
import types
from PIL import Image, ImageDraw
import numpy as np


def postprocess(self, detections: torch.Tensor, input_img: Image, visualize: bool = False):
    """Converts pytorch tensor into interpretable format

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.
    This model supports either outputting a list of bounding boxes of the format
    (x0, y0, w, h) or outputting a `PIL.Image` with the bounding boxes
    and (class name, class confidence, object confidence) indicated.

    Args:
        detections (Tensor): Output Tensor to postprocess
        input_img (PIL.Image): Original input image which has not been preprocessed yet
        visualize (bool): If True outputs image with annotations else a list of bounding boxes
    """
    img = np.array(input_img)
    detections = non_max_suppression(detections.clone().detach(), 80)

    print(detections)

    new_w, new_h, pad_w, pad_h = get_new_size_and_padding(input_img)

    # The amount of padding that was added
    #pad_w = max(img.shape[0] - img.shape[1], 0) * (new_w / max(img.shape))
    #pad_h = max(img.shape[1] - img.shape[0], 0) * (new_h / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = new_h - pad_h
    unpad_w = new_w - pad_w

    list_detections = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection[0].data.cpu().numpy()
        # Rescale coordinates to original dimensions
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_h // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_w // 2) / unpad_w) * img.shape[1]
        list_detections.append((x1, y1, box_h, box_w, conf, cls_conf, cls_pred))

    if visualize:
        img_out = input_img
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        for bbox in list_detections:
            x1, y1, x2, y2, conf, cls_conf, cls_pred = bbox
            ctx.rectangle([(x1, y1), (x1 + x2, y1 + y2)], outline=(255, 0, 0, 255), width=2)
            ctx.text((x1 + 5, y1 + 10),
                     text="{}, {:.2f}, {:.2f}".format(self.label_to_class(int(cls_pred)), cls_conf, conf))
        del ctx
        return img_out

    return list_detections


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # box_corner = prediction.new(prediction.shape)
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_new_size_and_padding(img: Image):
    min_side = 608
    max_side = 1024

    w, h = img.size
    smallest_side = min(h, w)
    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(h, w)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    new_h = int(round(h * scale))
    new_w = int(round((w * scale)))

    pad_w = 32 - new_h % 32
    pad_h = 32 - new_w % 32
    return new_w, new_h, pad_w, pad_h


def preprocess(self, x):
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.

    Args:
        x (list or PIL.Image): input image or list of images.
    """

    new_w, new_h, pad_w, pad_h = get_new_size_and_padding(x)

    input_transform = transforms.Compose([
        transforms.Resize((new_w, new_h)),
        transforms.Pad((pad_w//2, pad_h//2)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    if type(x) == list:
        out_tensor = None
        for elem in x:
            out = input_transform(elem).unsqueeze(0)
            if out_tensor is not None:
                torch.cat((out_tensor, out), 0)
            else:
                out_tensor = out
    else:
        out_tensor = input_transform(x).unsqueeze(0)

    print(out_tensor.shape)
    print(x.size)
    return out_tensor


def build():
    model = resnet50(num_classes=80, pretrained=False)

    model.load_state_dict(torch.load(os.path.join(os.path.realpath(os.path.dirname(__file__)), "weights.pth"),
                                     map_location=torch.device('cpu')))

    # model = RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu',
    #                  mode='CNA', res_scale=1, upsample_mode='upconv')
    # model.load_state_dict(torch.load(os.path.join(os.path.realpath(os.path.dirname(__file__)),
    #                                               "weights.pth")),
    #                       strict=True)

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    return model

