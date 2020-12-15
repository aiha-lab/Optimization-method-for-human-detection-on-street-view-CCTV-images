from argparse import Namespace
import json
import os
from pathlib import Path

import torch

from models.experimental import attempt_load
# from utils.datasets import create_dataloader
# from utils.general import non_max_suppression, scale_coords
from kh_utils.datasets import create_dataloader
from kh_utils.general import non_max_suppression, clip_coords
from utils.torch_utils import select_device, time_synchronized

import sys

ORIG_HEIGHT = 1080
ORIG_WIDTH = 1920

PROFILING = True  # set to False when we upload


class TimeTracker(object):

    def __init__(self):
        self.t = time_synchronized() if PROFILING else 0.0

    def update(self):
        if not PROFILING:
            return 0.0
        new_t = time_synchronized()
        duration = new_t - self.t
        self.t = new_t
        return duration


def test(data,
         weights=None,
         batch_size=32,
         img_size=640,
         max_det=20,
         conf_thres=0.001,
         iou_thres=0.5,  # for NMS
         augment=False,  # eval mode
         num_threads=8,
         num_queue=8,
         start_time = 0
         ):
    #start_time = time_synchronized()
    tracker = TimeTracker()

    # ================================================================================  #
    # Set Model
    # ================================================================================  #

    # Initialize/load model and set device
    device = select_device(opt.device, batch_size=batch_size)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # Half inference
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()

    model_set_duration = tracker.update()

    # ================================================================================  #
    # Set DataLoader
    # ================================================================================  #

    # Dataloader
    path = opt.data
    # dataloader = create_dataloader(path, img_size, batch_size, model.stride.max())
    dataloader = create_dataloader(path, (int(img_size * ORIG_HEIGHT / ORIG_WIDTH), img_size),
                                   batch_size, model.stride.max(),
                                   num_workers=num_threads, queue_depth=num_queue)

    dataloader_set_duration = tracker.update()

    # ================================================================================  #
    # Inference
    # ================================================================================  #

    json_result = {
        'annotations': [],
        'param_num': None,
        'inference_time': None
    }

    data_load_duration = 0.0
    data_preprocess_duration = 0.0
    model_forward_duration = 0.0
    nms_duration = 0.0
    post_process_duration = 0.0

    with torch.no_grad():
        _ = tracker.update()  # start for loop

        # for batch_i, (img, _, paths, shapes) in enumerate(dataloader):
        for batch_i, (img, paths) in enumerate(dataloader):
            data_load_duration += tracker.update()

            # --------------------------------------------------------------------------------  #
            # Data Preprocess
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            # img /= 255.0  # [0, 255] to [0.0, 1.0] already done inside
            nb, _, height, width = img.shape  # batch size, channels, height, width

            data_preprocess_duration += tracker.update()

            # --------------------------------------------------------------------------------  #
            # Model Forward
            inf_out, _ = model(img, augment=augment)  # inference and training outputs (FP16, GPU)
            model_forward_duration += tracker.update()

            # --------------------------------------------------------------------------------  #
            # NMS

            # output = non_max_suppression(inf_out, max_det=max_det,
            #                              conf_thres=conf_thres, iou_thres=iou_thres,
            #                              return_per_image=True)  # (FP32, GPU)
            output, out_indices = non_max_suppression(inf_out, max_det=max_det,
                                                      conf_thres=conf_thres, iou_thres=iou_thres,
                                                      return_per_image=False)  # (FP32, GPU)
            nms_duration += tracker.update()

            # --------------------------------------------------------------------------------  #
            # Postprocess Results
            # scale_coords((height, width), output[:, :4], shapes[0][0], shapes[0][1])  # inplace change
            output[:, [0, 2]] *= (ORIG_WIDTH / width)
            output[:, [1, 3]] *= (ORIG_HEIGHT / height)
            clip_coords(output[:, :4], (ORIG_HEIGHT, ORIG_WIDTH))

            # tensor to list
            res = output.tolist()  # tolist op moves tensor from GPU to CPU
            indices = out_indices.tolist()
            assert len(res) == len(indices)

            # initialize annotation
            annotation_offset = len(json_result['annotations'])
            for image_idx in range(nb):
                path = Path(paths[image_idx])
                frame = {
                    'file_name': path.stem + ".jpg",
                    'objects': []
                }
                json_result['annotations'].append(frame)

            # put annotation
            for res_idx in range(len(indices)):
                image_idx = indices[res_idx]
                *xyxy, conf, cls = res[res_idx]  # [x1, y1, x2, y2, conf, cls] FP32
                position = [int(i) for i in xyxy]
                confidence_score = round(conf, 3)
                json_result['annotations'][annotation_offset + image_idx]['objects'].append({
                    'position': position,
                    'confidence_score': str(confidence_score)
                })

            post_process_duration += tracker.update()

        _ = tracker.update()  # end for loop

    # ================================================================================  #
    # Save JSON
    # ================================================================================  #
    

    json_path = os.path.join('./', 't4_res_U0000000221.json')
    with open(json_path, 'w') as f:
        json_result['param_num'] = sum(p.numel() for p in model.parameters())
        end_time = time_synchronized()
        json_result['inference_time'] = end_time - start_time
        json.dump(json_result, f, indent=2, separators=(',', ":"))
        print("t4_res_U0000000221.json saved")

    json_duration = tracker.update()

    if PROFILING:
        print("[TIME] Set Model:", model_set_duration)
        print("[TIME] Set DataLoader:", dataloader_set_duration)
        print("[TIME] Total Data Loading:", data_load_duration)
        print("[TIME] Total Data Preprocessing:", data_preprocess_duration)
        print("[TIME] Total Model Forward:", model_forward_duration)
        print("[TIME] Total NMS:", nms_duration)
        print("[TIME] Total Postprocessing:", post_process_duration)
        print("[TIME] Save JSON:", json_duration)
        print("[TIME] Final Score (Inference Time):", json_result['inference_time'])
    return json_result['inference_time']


if __name__ == '__main__':
    time = time_synchronized()
    print("=" * 48)
    print("Data Path:", sys.argv[1])
    opt = Namespace(
        device='',
        # weights = 'wm0.1.pt',
        weights='weights/wm0.05_960.pt',
        data=sys.argv[1],
        batch_size=32,
        img_size=960,  # TODO change to 640?
        max_det=30,
        conf_thres=0.008,  # TODO set
        iou_thres=0.5,
        num_threads=8,  # TODO tune DALI
        num_queue=8,  # TODO tune DALI
    )
    print(opt)
    print("=" * 48)

    #torch.backends.cudnn.benchmark = True

    # if opt.task in ['val', 'test']:  # run normally
    test(data=opt.data,
         weights=opt.weights,
         batch_size=opt.batch_size,
         img_size=opt.img_size,
         max_det=opt.max_det,
         conf_thres=opt.conf_thres,
         iou_thres=opt.iou_thres,
         num_threads=opt.num_threads,
         num_queue=opt.num_queue,
         start_time = time
         )
