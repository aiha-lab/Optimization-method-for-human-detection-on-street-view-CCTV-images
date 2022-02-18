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
YOLO model의 width multiplier 계수를 조절하는 방법을 통해 파라미터 갯수를 7,246,518개에서 **103,678개(98.6% pruning)**로 줄임.   

## 코드 실행 방법

```bash
bash iitp.sh
```

If you want to test other models, change weight path and image size in ```Namespace``` of ```predict.py``` (Line 201).
- Weight path: weights/wm0.05_960.pt (best score)  
- img_size = 960  
