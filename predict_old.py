import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

from utils.datasets import iitp_create_dataloader

import sys
import pdb

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.5,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None, # model?
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         log_imgs=0):  # number of logged images
    start_time = time_synchronized()
    # Initialize/load model and set device
    device = select_device(opt.device, batch_size=batch_size)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    load_time = time_synchronized()
    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
       model.half()

    # Configure
    model.eval()
    nc = 1

    # Dataloader
    #img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    #path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
    path = opt.data
    # dataloader, dataset = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)
    dataloader, dataset = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=False)
#    dataloader = iitp_create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]

    json_result = {'annotations' : [], 'param_num' : None, 'inference_time' : None}
    data_time = time_synchronized()
    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    for batch_i, (img, _, paths, shapes) in enumerate(dataloader):
        '''
        FIXME targets have to be removed
        '''
        t = time_synchronized()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t3 += time_synchronized() - t

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, _ = model(img, augment=augment)  # inference and training outputs
            
            t0 += time_synchronized() - t

            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=[])
            t1 += time_synchronized() - t

        t = time_synchronized()
        # Statistics per image
        for si, pred in enumerate(output):
            path = Path(paths[si])

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
            ''' result json '''
            frame = {'file_name' : None, 'objects' : []}
            frame['file_name'] = path.stem + '.jpg'

            for *xyxy, conf, cls in predn.tolist():
                if(conf != 0):
                    position = xyxy
                    position = [int (i) for i in position]
                    confidence_score = round(conf, 3)
                    frame['objects'].append({"position" : position, "confidence_score" : str(confidence_score)})
    
            json_result['annotations'].append(frame)

            ''' result json '''
        t2 += time_synchronized() - t

    print("PROFILING:",t3,t0,t1,t2)
    forward_time = time_synchronized()

    json_path = os.path.join('./','t4_res_U0000000221.json')
    with open(json_path, 'w') as f:
        json_result['param_num'] = sum(p.numel() for p in model.parameters())
        json_result['inference_time'] = (time_synchronized() - start_time)
        json.dump(json_result, f, indent = 2, separators = (',', ":"))
        print("t4_res_U0000000221.json saved")

    json_time = time_synchronized()

    print("load:", load_time-start_time)
    print("data:", data_time-load_time)
    print("forward:", forward_time-data_time)
    print("json:", json_time-forward_time)
    print("total:", json_time-start_time)
    return forward_time



if __name__ == '__main__':

    print(sys.argv[1])
    opt = Namespace(
        augment = False,
        #weights = 'wm0.1.pt',
        weights = 'weights/wm0.05_960.pt',
        data = sys.argv[1],
        batch_size = 32,
        img_size = 960,
#        img_size = 640,
        iou_thres = 0.5,
        name='exp',
        project = 'runs/test',
        device = '',
        exist_ok = True,
        task = 'val',
        single_cls = True
    )
    print(opt)

    #if opt.task in ['val', 'test']:  # run normally
    test(data=opt.data,
         weights=opt.weights,
         batch_size=opt.batch_size,
         imgsz=opt.img_size,
         iou_thres=opt.iou_thres,
         augment=opt.augment
         )
