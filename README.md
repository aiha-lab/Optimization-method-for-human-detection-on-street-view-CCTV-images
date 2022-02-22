# 외부 CCTV 거리 영상의 인간 검출 최적화 기술
본 프로그램의 특징: 본 프로그램은 최신 객체 검출 딥러닝 모델인 YOLOv5s를 활용하여, 거리가 촬영된 CCTV 영상에서 사람을 찾아 위치를 표기한다. 딥러닝 모델의 성능이 올라감에 따라 연산량도 굉장히 많아지게 되었는데, 이렇게 많은 연산량은 추론 속도를 저해하고 파워나 에너지를 더 많이 소모하기 때문에 최근에는 성능을 유지하면서 딥러닝 모델을 경량화하는 연구가 진행되고 있다. 본 프로그램은 YOLO 모델의 width multiplier를 조절하여 모델 파라미터 수를 98.6% 줄이면서도 (기존 모델의 1.4% 크기) CCTV 영상에서 사람을 효과적으로 찾아내도록 하였다.

주요 기능: 본 프로그램의 주요 기능은 FHD(1920*1080)의 영상에서 사람을 찾아 위치를 표기하는 것이다. 미리 학습되어 저장된 딥러닝 모델을 불러와 주어진 영상을 모델에 적합한 크기로 전처리하고, 모델을 통과시켜 특징을 추출한다. 추출된 특징을 기반으로 충분히 사람이라고 추측되는 객체에 박스를 그려 표기하며, 겹치는 박스들에 대해서는 점수가 높은 것을 기준으로 최적의 박스를 하나 결정한다. 마지막으로 추출된 특징에서의 박스가 원본 영상에서 실제 어디에 위치하는지 판단하여, 영상 속에 사람이라고 생각되는 객체에 박스를 표기한다.

사용방법: 프로그램은 Python 언어를 사용한 딥러닝 프레임워크인 PyTorch를 사용하여 작성하였다. 따라서 PyTorch와 python이 기본적으로 설치된 환경이 요구된다. 실행 방법은 bash 환경 스크립트 파일인 .sh 로 작성되었으며, 'bash iitp.sh'로 실행 가능하다. 실행하게 되면 예제 검증 데이터셋에 대해서 사람을 검출하고, 박스의 위치를 json 파일 형태로 저장하게 된다. 또한 predict.py 안에 Namespace로 정의된 여러가지 하이퍼파라미터나, 미리 학습된 모델 등을 변경할 수 있다.

# 2020 인공지능 그랜드 챌린지  
주최: 과학기술정보통신부, 정보통신기획평가원  
임무: 주어진 학습데이터를 기반으로 베이스라인 대비 성능은 유지하되, 사이즈와 연산속도가 향상된 모델을 제시하라  

# Baseline model  
YOLOv5s (https://github.com/ultralytics/yolov5)  
Parameter 갯수: 7,246,518

# Dataset & Task
Dataset: CCTV로 촬영된 FHD(1920x1080) 영상    
Task: 사람 검출 (Human detection)  

## Dataset 예시 

![0506_V0001_008](https://user-images.githubusercontent.com/60534494/154604685-a878dd75-80d6-45d1-ab1b-d15d8349571d.jpg)

![0608_V0014_107](https://user-images.githubusercontent.com/60534494/154604701-80d31165-d720-45dd-b84f-6c5768aef8e8.jpg)

![0601_V0004_274](https://user-images.githubusercontent.com/60534494/154604706-6827ab34-f1e9-4a5f-874d-ccd2ab67cd44.jpg)

# Performance  

## Detection 예시 및 mAP

![d0506_V0001_008](https://user-images.githubusercontent.com/60534494/154605151-082e8be8-97af-4db8-a6fc-0f9e893977f0.jpg)

![d0608_V0014_107](https://user-images.githubusercontent.com/60534494/154605170-b389dd10-cc47-40cd-b0df-54fd2bd02935.jpg)

![d0601_V0004_274](https://user-images.githubusercontent.com/60534494/154605185-42c93982-85fe-45bf-a030-2bb6e1dcbe6d.jpg)

사람이 있는 곳에 Bounding Box를 잘 표기하는 것을 볼 수 있음.  
대회의 test dataset에 대해 **72.9**의 mAP를 달성함.  

## 모델 경량화
YOLO model의 width multiplier 계수를 조절하는 방법을 통해 파라미터 갯수를 7,246,518개에서 **103,678개**(98.6% pruning)로 줄임.   

## 코드 실행 방법
Reference code: https://github.com/ultralytics/yolov5  
Contributors: Kyuhong Shim(@SNU), Minsoo Kim(@HYU), Janghwan Lee(@HYU)  
This repo contains inference code for YOLOv5.  

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

<table>
<tr>
<td>

Before optimization
</td>
<td>

Before num-thread tuning
</td>
<td> 

Final optimization
</td>
</tr>
<tr>
<td>

```bash
[TIME] Set Model: 0.4797539710998535
[TIME] Set DataLoader: 0.6640267372131348
[TIME] Total Data Loading: 21.1922824382782
[TIME] Total Data Preprocessing: 1.0619120597839355
[TIME] Total Model Forward: 3.369990825653076
[TIME] Total NMS: 10.082083225250244
[TIME] Total Postprocessing: 6.487901926040649
[TIME] Save JSON: 3.122708320617676
[TIME] Final Score (Inference Time): 46.06066
```
</td>
<td>

```bash
[TIME] Set Model: 0.4932844638824463
[TIME] Set DataLoader: 1.6099259853363037
[TIME] Total Data Loading: 10.260902881622314
[TIME] Total Data Preprocessing: 0.07654070854187012
[TIME] Total Model Forward: 4.588696241378784
[TIME] Total NMS: 0.9312660694122314
[TIME] Total Postprocessing: 0.9399559497833252
[TIME] Save JSON: 1.6605088710784912
[TIME] Final Score (Inference Time): 18.900789499282837
```
</td>
<td>

```bash
[TIME] Set Model: 0.4765970706939697
[TIME] Set DataLoader: 1.599604845046997
[TIME] Total Data Loading: 3.299018144607544
[TIME] Total Data Preprocessing: 0.07267594337463379
[TIME] Total Model Forward: 4.18438982963562
[TIME] Total NMS: 0.8949720859527588
[TIME] Total Postprocessing: 0.8590140342712402
[TIME] Save JSON: 1.6375606060028076
[TIME] Final Score (Inference Time): 11.386500835418701
```
</td>
</tr>
</table>

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

![image](https://user-images.githubusercontent.com/60534494/102202640-ecccce00-3f0a-11eb-9afa-3cfb0df0eabb.png)


### 3. Hardware Parameters

We check that ```batch_size=32``` and ```num_threads=8``` is the best condition on V100.  

![스크린샷 2020-12-12 오전 11 45 09](https://user-images.githubusercontent.com/60534494/102201017-eb9aa180-3f08-11eb-8456-bf30e0226564.png)

https://user-images.githubusercontent.com/60534494/102200832-ae361400-3f08-11eb-8a1a-7cb5c6910cbe.png

![스크린샷 2020-12-12 오전 11 52 41](https://user-images.githubusercontent.com/60534494/102201098-fe14db00-3f08-11eb-8879-c4255ec77e53.png)

https://user-images.githubusercontent.com/60534494/102200883-bee68a00-3f08-11eb-9810-82cb3de7a034.png

## About Codes  

### Dataloader with NVIDIA DALI for speedup  
```py
#kh_utils/datasets.py
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes

def create_dataloader(path, image_size, batch_size, stride, device_id=0, num_workers=8, queue_depth=8):
    pipeline = ImageFolderValidPipe(batch_size, num_workers, device_id, path, image_size, stride,
                                    mean=0.0, stddev=255.0, queue_depth=queue_depth)
    pipeline.build()
    loader = DALIValidIterator(pipeline)
    return loader
```
NVIDIA DALI reduces the time for data loading 17.46s to 0.25s

### NMS + Postprocessing
```py
#predict.py
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
clip_coords(output[:, :4], (ORIG_HEIGHT, ORIG_WIDTH)
```
Because the whole datasets' image size is same, we scale-down the bbox at a once.  
This optimization reduces the time for nms+bbox-labeling 17.59s to 2.13s


# LICENSE
GNU General Public License v3.0
