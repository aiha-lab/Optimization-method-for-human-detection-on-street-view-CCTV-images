import time
import numpy as np
import torch
import torchvision


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[..., [0, 2]] -= pad[0]  # x padding
    coords[..., [1, 3]] -= pad[1]  # y padding
    coords[..., :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[..., 0].clamp_(0, img_shape[1])  # x1
    boxes[..., 1].clamp_(0, img_shape[0])  # y1
    boxes[..., 2].clamp_(0, img_shape[1])  # x2
    boxes[..., 3].clamp_(0, img_shape[0])  # y2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.copy(x)
    w_half = x[..., 2] / 2
    h_half = x[..., 3] / 2
    y[..., 0] = x[..., 0] - w_half  # top left x
    y[..., 1] = x[..., 1] - h_half  # top left y
    y[..., 2] = x[..., 0] + w_half  # bottom right x
    y[..., 3] = x[..., 1] + h_half  # bottom right y
    return y


def non_max_suppression(prediction,
                        conf_thres=0.1, iou_thres=0.5,
                        *, max_det=20, agnostic=False, return_per_image=True):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Params:
        prediction: (batch_size, locations, 4 + 1 + num_classes)

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    batch_size = prediction.shape[0]
    num_classes = prediction.shape[-1] - 5  # number of classes

    # --------------------------------------------------------------------------------  #
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    multi_label = (num_classes > 1)  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.float()  # crucial detail!

    # --------------------------------------------------------------------------------  #
    # Filter by object-ness
    object_candidate = (prediction[..., 4] > conf_thres).nonzero(as_tuple=False).T  # candidates (2, nonzero_targets)
    # object_candidate[0]: batch_idx, object_candidate[1]: location_idx
    prediction = prediction[object_candidate[0], object_candidate[1], :].contiguous()
    # prediction: (num_targets, 4 + 1 + num_classes)
    if not prediction.shape[0]:  # no targets
        return [torch.zeros(0, 6, dtype=prediction.dtype, device=prediction.device)] * batch_size

    # prediction = torch.cat((prediction, object_candidate[0][:, None]), dim=1)
    # prediction: (num_targets, 4 + 1 + num_classes + 1)

    # --------------------------------------------------------------------------------  #
    # Prepare NMS
    prediction[:, 5:] *= prediction[:, 4:5]  # conf = obj_conf * cls_conf
    box = xywh2xyxy(prediction[..., :4])  # (center_x, center_y, width, height) -> (x1, y1, x2, y2)

    if multi_label:
        conf_candidate = (prediction[:, 5:] > conf_thres).nonzero(as_tuple=False).T  # candidates (2, num_targets)
        # conf_candidate[0]: location_idx, conf_candidate[1]: class_idx - 5
        boxes = box[conf_candidate[0]]  # (num_targets, 4)
        scores = prediction[conf_candidate[0], conf_candidate[1] + 5]  # (num_targets,)
        classes = conf_candidate[1].float()  # (num_targets,)
        indices = object_candidate[0][conf_candidate[0]]  # (num_targets,)
    else:  # always class 0
        conf_candidate = (prediction[:, 5] > conf_thres).nonzero(as_tuple=False).T  # candidates (1, num_targets,)
        # conf_candidate[0]: location_idx
        boxes = box[conf_candidate[0]]  # (num_targets, 4)
        scores = prediction[conf_candidate[0], 5]  # (num_targets,)
        classes = torch.zeros(boxes.shape[0], dtype=boxes.dtype, device=boxes.device)  # (num_targets,)
        indices = object_candidate[0][conf_candidate[0]]  # (num_targets,)

    if not boxes.shape[0]:  # no targets
        return [torch.zeros(0, 6, dtype=prediction.dtype, device=prediction.device)] * batch_size

    # --------------------------------------------------------------------------------  #
    # Run batched NMS
    offset = classes * (0 if agnostic else max_wh)
    boxes_with_offset = boxes + offset[:, None]  # boxes (offset by class)

    nms_indices = torchvision.ops.boxes.batched_nms(boxes_with_offset, scores, indices, iou_threshold=iou_thres)
    nms_result_indices = indices[nms_indices]  # (num_boxes,)

    nms_result = torch.cat([boxes[nms_indices, :],
                            scores[nms_indices, None],
                            classes[nms_indices, None]], dim=1)  # (num_boxes, 6)

    # --------------------------------------------------------------------------------  #
    # Split to per-batch (most slowest part)

    if return_per_image:
        output = []
        for image_idx in range(batch_size):
            image_result = nms_result[nms_result_indices == image_idx].clone()
            if image_result.shape[0] > max_det:
                image_result = image_result[:max_det]
            output.append(image_result)
    else:
        output = (nms_result, nms_result_indices)

    return output
