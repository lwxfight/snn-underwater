# snn-underwater

This is the code about the paper:``SU-YOLO: Spiking Neural Network for Efficient Underwater Object Detection.''

Underwater object detection is critical for oceanic research and industrial safety inspections. However, the complex optical environment and the limited resources of underwater equipment pose significant challenges to achieving high accuracy and low power consumption. To address these issues, we propose Spiking Underwater YOLO (SU-YOLO), a Spiking Neural Network (SNN) model. Leveraging the lightweight and energy-efficient properties of SNNs, SU-YOLO incorporates a novel spike-based underwater image denoising method based solely on integer addition, which enhances the quality of feature maps with minimal computational overhead. In addition, we introduce Separated Batch Normalization (SeBN), a technique that normalizes feature maps independently across multiple time steps and is optimized for integration with residual structures to capture the temporal dynamics of SNNs more effectively. The redesigned spiking residual blocks integrate the Cross Stage Partial Network (CSPNet) with the YOLO architecture to mitigate spike degradation and enhance the model's feature extraction capabilities. Experimental results on URPC2019 underwater dataset demonstrate that SU-YOLO achieves mAP of 78.8% with 6.97M parameters and an energy consumption of 2.98 mJ, surpassing mainstream SNN models in both detection accuracy and computational efficiency. These results underscore the potential of SNNs for engineering applications. 

![image](https://github.com/lwxfight/snn-underwater/blob/main/su-yolo.png)

# Instructions(V1)

# Requirements and Dependencies






## Training：

1. Download the source videos from 
2. Training

  coming soon

## Testing:

Specifically, we need to create our new config for testing and run multi-crop/multi-clip test:

1. Copy the training config file `config.yaml` and create new testing config `test.yaml`.

2. Change the hyperparameters of data (in `test.yaml`):

   ```yaml
   DATA:
     TRAIN_JITTER_SCALES: [224, 224]
     TEST_CROP_SIZE: 224
   ```

3. Set the number of crops and clips (in `test.yaml`):

   **Multi-clip testing for Kinetics**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 4
   TEST.NUM_SPATIAL_CROPS 1
   ```

   **Multi-crop testing for Something-Something**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 1
   TEST.NUM_SPATIAL_CROPS 3
   ```

4. You can also set the checkpoint path via:

   ```shell
   TEST.CHECKPOINT_FILE_PATH your_model_path
   ```

5. An example:

```

```



# Your Own Dataset：

1. Prepare data
2. Training
3. Coming soon
