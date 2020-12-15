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

## Checklist

### 1. About Code Optimization

|             | Before optimization                     | After optimization (Submitted)                               |
|-------------|-----------------------------------------|---------------------------------------------------|
| Main        | ```predict_old.py```                          | ```predict.py```                                        |
| Data loader | ```utils/datasets.py#336```                   | ```kh_utils/datasets.py#90```<br>                           |
| NMS         | ```utils/general.py#260```<br>(nms: line 331) | ```kh_utils/general.py#52```<br>(batched nms: line 113) |

You can run old version by the command ```bash scripts/predict_old.sh```  

### 2. About Time Checking  

<table>
<tr>
<td> Command </td>
<td>

```bash iitp.sh``` (submitted)
</td>
<td> 

```bash scripts/predict_chk.sh``` 
</td>
</tr>
<tr>
<td> Result </td>
<td>

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
[TIME] Set Model: 0.48934483528137207
[TIME] Set DataLoader: 1.6480441093444824
[TIME] Total Data Loading: 2.8038485050201416
[TIME] Total Data Preprocessing: 0.13748860359191895
[TIME] Total Model Forward: 4.888248443603516
[TIME] Total NMS: 1.5890100002288818
[TIME] Total Postprocessing: 0.8877518177032471
[TIME] Save JSON: 1.648402214050293
[TIME] Final Score (Inference Time): 12.44399905204773
[TOTAL PARAMS] Total Num of Model Parameters: 103678
Predict Done.........
IITP mAP caclulate Start.........
AP Score: 0.7245907193874522
IITP mAP caclulate Done.........
Remove all json file
```
</td>
<td>

```bash
Predict Start........
================================================
Data Path: /raid/iitpdata/images/val/
Namespace(batch_size=32, conf_thres=0.008, data='/raid/iitpdata/images/val/', device='', img_size=960, iou_thres=0.5, max_det=30, num_queue=8, num_threads=8, weights='weights/wm0.05_960.pt')
================================================
Fusing layers... 
Resetting DALI loader
t4_res_U0000000221.json saved
[TIME] Set Model: 0.48699355125427246
[TIME] Set DataLoader: 1.6262810230255127
[TIME] Total Data Loading: 3.059208631515503
[TIME] Total Data Preprocessing: 0.14332365989685059
[TIME] Total Model Forward: 4.6812005043029785
[TIME] Total NMS: 1.2890410423278809
[TIME] Total Postprocessing: 0.8272891044616699
[TIME] Save JSON: 1.6297228336334229
[TIME] Final Score (Inference Time): 12.115106344223022
Predict Done.........
IITP mAP caclulate Start.........
AP Score: 0.7245907193874522
IITP mAP caclulate Done.........
Remove all json file
```
</td>
</tr>
<tr>
<td> Main file </td> 
<td> 

```predict.py``` 
</td> 
<td> 

```predict_chk.py``` 
</td> 
</tr>
</table>

### 3. Hardware Parameters

We check that ```batch_size=32``` and ```num_threads=8``` is the best condition on V100.  

![스크린샷 2020-12-12 오전 11 45 09](https://user-images.githubusercontent.com/60534494/102201017-eb9aa180-3f08-11eb-8456-bf30e0226564.png)

https://user-images.githubusercontent.com/60534494/102200832-ae361400-3f08-11eb-8a1a-7cb5c6910cbe.png

![스크린샷 2020-12-12 오전 11 52 41](https://user-images.githubusercontent.com/60534494/102201098-fe14db00-3f08-11eb-8879-c4255ec77e53.png)

https://user-images.githubusercontent.com/60534494/102200883-bee68a00-3f08-11eb-9810-82cb3de7a034.png

