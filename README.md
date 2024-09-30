## Train
You can use the following command to perform the training.
```bash
python train_seg.py --data_dir {data/train} --BATCH_SIZE 20 --NUM_EPOCHS 10
```
The training set is in the form of:
```bash
train
  └── neg
  └── pos
```
## Obtaining Pre-trained Model Weights
To facilitate the use of pre-trained models in this project, we have hosted the pre-trained weights file pretrain.pth on a web storage service. You can download the file from the following link:
https://drive.google.com/file/d/1B9gxeAiWcFvTeQbSGmSvDL09Hu-uL5Z9/view?usp=drive_link
