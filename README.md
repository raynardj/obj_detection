# Ray's Object Detection Framework

## A pytorch deployment
## On yolo v3 style model

### Modules:

* obj_detection.ipynb - The trunk line experimental notebook
    * For now it's DenseNet121 (Very LightWeight, resilient to over-fitting)
* constant.py - configurations
* data.py -  data generator for obj detection
* loss_.py - loss function
* utils.py - tools and other functions
* conv_modle.py - feature extractor

### Reference:

1. [You Only Look Once: Unified, Real-Time Object Detection](http://arxiv.org/abs/1506.02640)

2. [YOLO9000:
Better, Faster, Stronger](http://arxiv.org/abs/1612.08242)

3. [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

4. [R-FCN: Object Detection via
Region-based Fully Convolutional Networks](http://arxiv.org/abs/1605.06409)

5. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)