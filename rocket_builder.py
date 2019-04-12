import os
from .architecture import resnet50
import torch
import torch.nn as nn
from torchvision import transforms
import types
from PIL import Image, ImageDraw
import numpy as np
import math
from .architecture import ClassificationModel

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def label_to_class(self, label: int) -> str:
    """Returns string of class name given index
    """
    return self.classes[label]

def clamp(n, minn, maxn):
    """Make sure n is between minn and maxn

    Args:
        n (number): Number to clamp
        minn (number): minimum number allowed
        maxn (number): maximum number allowed
    """
    return max(min(maxn, n), minn)

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
    img_height, img_width, _ =  img.shape

    detections = non_max_suppression(detections.clone().detach(), 80)[0]

    new_w, new_h, pad_w, pad_h, scale = get_new_size_and_padding(input_img)

    # The amount of padding that was added
    #pad_w = max(img.shape[0] - img.shape[1], 0) * (new_w / max(img.shape))
    #pad_h = max(img.shape[1] - img.shape[0], 0) * (new_h / max(img.shape))
    # Image height and width after padding is removed


    unpad_h = new_h
    unpad_w = new_w

    list_detections = []
    for detection in detections:
        # print(detection.int())
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection.data.cpu().numpy()
        # # Rescale coordinates to original dimensions
        # box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        # box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        # y1 = ((y1 - pad_h // 2) / unpad_h) * img.shape[0]
        # x1 = ((x1 - pad_w // 2) / unpad_w) * img.shape[1]

        # list_detections.append((x1, y1, box_h, box_w, conf, cls_conf, cls_pred))

        # Rescale coordinates to original dimensions
        x1 = ((x1 - pad_w // 2) / unpad_w) * img_width
        y1 = ((y1 - pad_h // 2) / unpad_h) * img_height

        x2 = ((x2 - pad_w // 2) / unpad_w) * img_width
        y2 = ((y2 - pad_h // 2) / unpad_h) * img_height
        
        # Standardize the output
        topLeft_x = int(clamp(round(x1), 0, img_width))
        topLeft_y = int(clamp(round(y1), 0, img_height))

        bottomRight_x = int(clamp(round(x2), 0, img_width))
        bottomRight_y = int(clamp(round(y2), 0, img_height))

        width = abs(bottomRight_x - topLeft_x) + 1
        height = abs(bottomRight_y - topLeft_y) + 1

        bbox_confidence = clamp(conf, 0, 1)

        class_name = str(self.label_to_class(int(cls_pred)))
        class_confidence = clamp(cls_conf, 0, 1)

        list_detections.append({
                'topLeft_x': topLeft_x,
                'topLeft_y': topLeft_y,
                'width': width,
                'height': height,
                'bbox_confidence': bbox_confidence,
                'class_name': class_name,
                'class_confidence': class_confidence})

    if visualize:
        line_width = 2
        img_out = input_img
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        for detection in list_detections:
            # Extract information from the detection
            topLeft = (detection['topLeft_x'], detection['topLeft_y'])
            bottomRight = (detection['topLeft_x'] + detection['width'] - line_width, detection['topLeft_y'] + detection['height']- line_width)
            class_name = detection['class_name']
            bbox_confidence = detection['bbox_confidence']
            class_confidence = detection['class_confidence']

            # Draw the bounding boxes and the information related to it
            ctx.rectangle([topLeft, bottomRight], outline=(255, 0, 0, 255), width=line_width)
            ctx.text((topLeft[0] + 5, topLeft[1] + 10), text="{}, {:.2f}, {:.2f}".format(class_name, bbox_confidence, class_confidence))

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
        # print(image_i, image_pred)
        # print(image_pred.shape)
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

    return new_w, new_h, pad_w, pad_h, scale


def train_forward(self, x: torch.Tensor, targets: torch.Tensor):
    """Performs forward pass and returns loss of the model

    The loss can be directly fed into an optimizer.
    """
    self.training = True
    loss = self.forward((x, targets))
    self.training = False
    return loss


def preprocess(self, img: Image, labels: list = None) -> torch.Tensor:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """

    # todo: support batch size bigger than 1 for training and inference
    # todo: replace this hacky solution and work directly with tensors
    if type(img) == Image.Image:
        # PIL.Image
        pass
    elif type(img) == torch.Tensor:
        # list of tensors
        img = img[0].cpu()
        img = transforms.ToPILImage()(img)
    elif "PIL" in str(type(img)): # type if file just has been opened
        img = img.convert("RGB")
    else:
        raise TypeError("wrong input type: got {} but expected list of PIL.Image, "
                        "single PIL.Image or torch.Tensor".format(type(img)))

    new_w, new_h, pad_w, pad_h, _ = get_new_size_and_padding(img)

    # print(x.size, new_w, new_h)

    input_transform = transforms.Compose([
        transforms.Resize((new_h, new_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    padding = torch.nn.ConstantPad2d((pad_h//2, pad_h//2, pad_w//2, pad_w//2), 0.0)

    out_tensor = input_transform(img).unsqueeze(0)
    out_tensor = padding(out_tensor)

    if labels is None:
        return out_tensor

    # if type(img) == list:
    #     out_tensor = None
    #     for elem in img:
    #         out = input_transform(elem).unsqueeze(0)
    #         if out_tensor is not None:
    #             torch.cat((out_tensor, out), 0)
    #         else:
    #             out_tensor = out
    # else:
    #     out_tensor = input_transform(img).unsqueeze(0)

    max_objects = 50
    filled_labels = np.zeros((max_objects, 5))  # max objects in an image for training=50, 5=(x1,y1,x2,y2,category_id)
    if labels is not None:
        for idx, label in enumerate(labels):

            # add padding
            label[0] += pad_w//2
            label[1] += pad_h//2
            label[2] += pad_w // 2
            label[3] += pad_h // 2

            padded_w = new_w + pad_w
            padded_h = new_h + pad_h

            # resize coordinates to match Yolov3 input size
            scale_x = new_w / padded_w
            scale_y = new_h / padded_h

            label[0] *= scale_x
            label[1] *= scale_y
            label[2] *= scale_x
            label[3] *= scale_y

            x1 = label[0]
            y1 = label[1]
            x2 = label[2]
            y2 = label[3]

            # x1 = label[0] / new_w
            # y1 = label[1] / new_h
            #
            # cw = (label[2]) / new_w
            # ch = (label[3]) / new_h
            #
            # cx = (x1 + (x1 + cw)) / 2.0
            # cy = (y1 + (y1 + ch)) / 2.0

            filled_labels[idx] = np.asarray([x1, y1, x2, y2, label[4]])
            if idx >= max_objects:
                break
    filled_labels = torch.from_numpy(filled_labels)

    return out_tensor, filled_labels.unsqueeze(0)


def rebuild_head(self, num_classes):
    """Rebuilds layers needed to train on dataset with different amount of classes

    Use this method to adapt the network structure to a new dataset.
    """
    self.num_classes = num_classes
    device = "cpu" if not next(self.classificationModel.parameters()).is_cuda else "cuda"

    classifier = ClassificationModel(256, num_classes=num_classes)
    classifier.output.weight.data.fill_(0)
    prior = 0.01
    classifier.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

    self.classificationModel = classifier
    self.to(device)



def build():
    model = resnet50(num_classes=80, pretrained=False)

    model.load_state_dict(torch.load(os.path.join(os.path.realpath(os.path.dirname(__file__)), "weights.pth"),
                                     map_location=torch.device('cpu')))

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    classes = load_classes(os.path.join(os.path.realpath(os.path.dirname(__file__)), "coco.data"))

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    model.label_to_class = types.MethodType(label_to_class, model)
    model.train_forward = types.MethodType(train_forward, model)
    model.rebuild_head = types.MethodType(rebuild_head, model)
    setattr(model, 'classes', classes)

    return model

