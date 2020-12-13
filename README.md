# AI Challenge 4th - Track 4 (Dec 2020)

Reference code: https://github.com/ultralytics/yolov5  
Contributors: Kyuhong Shim(@SNU), Minsoo Kim(@HYU), Janghwan Lee(@HYU)  
This repo contains inference code for YOLOv5.  

## Commands and infos   

```bash
bash iitp.sh
```

If you want to test other models, change weight path and image size in ```Namespace``` of ```predict.py``` (Line 201).
- Weight path: weights/wm0.05_960.pt (best score)  
- img_size = 960  

## Example output  
```bash
Predict Start........
================================================
Data Path: /raid/iitpdata/images/val/
Namespace(batch_size=32, conf_thres=0.008, data='/raid/iitpdata/images/val/', device='', img_size=960, iou_thres=0.5, max_det=30, num_queue=8, num_threads=8, weights='weights/wm0.05_960.pt')
================================================
Fusing layers... 
Resetting DALI loader
Image shape: torch.Size([6, 3, 544, 960])
t4_res_U0000000221.json saved
[TIME] Set Model: 0.5013864040374756
[TIME] Set DataLoader: 1.6690936088562012
[TIME] Total Data Loading: 3.0299623012542725
[TIME] Total Data Preprocessing: 0.13550972938537598
[TIME] Total Model Forward: 4.620744705200195
[TIME] Total NMS: 1.4857075214385986
[TIME] Total Postprocessing: 0.8144493103027344
[TIME] Save JSON: 1.6711077690124512
[TIME] Final Score (Inference Time): 12.257069826126099
[TOTAL PARAMS] Total Num of Model Parameters: 103678
Predict Done.........
IITP mAP caclulate Start.........
AP Score: 0.7245907193874522
IITP mAP caclulate Done.........
Remove all json file
```
