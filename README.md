# Soybean-Flowers-Detection
This project introduces a method based on Faster RCNN to detect the number of soybean flowers to solve the problem of manual counting of soybean flowers. Help breeders understand the genetic mechanism of flower drop and increase soybean yield. The project code comes from the official website of [MxNet](https://cv.gluon.ai/build/examples_detection/train_faster_rcnn_voc.html#sphx-glr-build-examples-detection-train-faster-rcnn-voc-py), modified to be used for soybean flower count problem. The picture shows the overall flow chart of the experiment.![image](https://github.com/yanzhuangzhuang123/Soybean-Flowers-Detection/tree/main/dataset/flow.png)
# Install
Please refer to the official installation tutorial [MxNet Installation](https://cv.gluon.ai/install/install-more.html#)
## Required environment
* python>=3.6.5
* MxNet>=1.6
* Anaconda3 (recommended)
# Training
Train a default resnet50_v1b model with Pascal VOC on GPU 0:
```python
python train_faster_rcnn.py --gpus 0
``` 
Train a resnet50_v1b model on GPU 0,1,2,3:
```python
python train_faster_rcnn.py --gpus 0,1,2,3 --network resnet50_v1b
``` 
Check the supported arguments:
```python
python train_faster_rcnn.py --help
``` 
# TEST
Model evaluation
```python
python eval_fasterrcnn.py 
```
# Predict 
```python
python Predict.py
```
# The following figure shows the effect of model detection
![image](https://github.com/yanzhuangzhuang123/Soybean-Flowers-Detection/tree/main/dataset/result.png)

# DATASET
Download link of soybean flower dataset: https://pan.baidu.com/s/1pwYNLUUkG9ueayUrm6sZBw.
Password：le80 .
If you want to use this dataset, please add the corresponding reference or contact us.
