# SU-YOLO: Spiking Neural Network for Efficient Underwater Object Detection [[Neurocomputing]](https://www.sciencedirect.com/science/article/pii/S0925231225009828?dgcid=coauthor)

[Chenyang Li](), [Wenxuan Liu](), [Guoqiang Gong](), [Xiaobo Ding]() and [Xian Zhong]()


## Abstract

Underwater object detection is critical for oceanic research and industrial safety inspections. However, the complex optical environment and the limited resources of underwater equipment pose significant challenges to achieving high accuracy and low power consumption. To address these issues, we propose Spiking Underwater YOLO (SU-YOLO), a Spiking Neural Network (SNN) model. Leveraging the lightweight and energy-efficient properties of SNNs, SU-YOLO incorporates a novel spike-based underwater image denoising method based solely on integer addition, which enhances the quality of feature maps with minimal computational overhead. In addition, we introduce Separated Batch Normalization (SeBN), a technique that normalizes feature maps independently across multiple time steps and is optimized for integration with residual structures to capture the temporal dynamics of SNNs more effectively. The redesigned spiking residual blocks integrate the Cross Stage Partial Network (CSPNet) with the YOLO architecture to mitigate spike degradation and enhance the model's feature extraction capabilities. Experimental results on URPC2019 underwater dataset demonstrate that SU-YOLO achieves mAP of 78.8% with 6.97M parameters and an energy consumption of 2.98 mJ, surpassing mainstream SNN models in both detection accuracy and computational efficiency. These results underscore the potential of SNNs for engineering applications.

![](SU-YOLO.png)


## Requirements

```
python == 3.8 or 3.9
cuda >= 11.8
pytorch >= 2.0.0
torchvision>=0.15.1
numpy >= 1.21.0
spikingjelly >= 0.0.0.0.12
```


## Usage

#### Installation
```shell
pip install -r requirements.txt
```

#### Training

```shell
python train.py --workers 4 --device 0 --batch 16 --data ./data/urpc.yaml --img 320 --cfg models/detect/su-yolo.yaml --name any-name --epochs 300 --time-step 4
```

#### Testing

```shell
python val.py --device 0 --batch 16 --data ./data/urpc.yaml --img 320 --name any-name --weights './checkpoint.pt'
```


## Datasets

Please download in "YOLOv9" format.

- [URPC2019](https://universe.roboflow.com/underwater-fish-f6cri/urpc2019-nrbk1)

- [UDD](https://universe.roboflow.com/yuyutsu/udd-j8cpd)


## Checkpoints
- [Google Drive](https://drive.google.com/file/d/18pl_CDkm88momMpfn65ChOtsxUEf0xXb/view?usp=sharing)


## Citation

```
@article{LI2025130310,
title = {SU-YOLO: Spiking neural network for efficient underwater object detection},
journal = {Neurocomputing},
volume = {644},
pages = {130310},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.130310},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225009828},
author = {Chenyang Li and Wenxuan Liu and Guoqiang Gong and Xiaobo Ding and Xian Zhong},
}
```

For help or issues using this git, please submit a GitHub issue.

Thanks to [Chenyang Li](https://github.com/CaoJu600) for his great contribution to building this project.


For other communications related to this git, please contact `chenyang@ctgu.edu.cn` and `lwxfight@126.com`.

