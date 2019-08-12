# pcam_ai (SUTD AI Project)
### Members:

- Bernard Cheng (1002053)
- Du Wanping (1002386)
- Karthic Harish (1002265)
- You Song Shan (1002346)

---

### Project Introduction

The project aims to explore different neural networks, data augmentation methods and other training parameters to classify camelyon from histopathologic scans of lymph node sections. We also created a GUI as a platform to make prediction on any picture chosen by user using our pre-trained classifiers. In this project, we have explored Resnet50, and Densenet169. 

### Approach & Chosen Models

Due to the size and complexity of the dataset, a deep convolutional network would be instrumental in learning the nuanced features in cell images. Hence, we decided to capitalize on the pre-trained models available on Keras, namely Resnet50 and Densenet169. Instead of using the Imagenet weights available in the package, we experimented with trainable layers. After all, Imagenet weights were learned from images such as animals, people and nature, just to name a few, which are markedly different from cancer cell images. Moreover, we included pooling and dropout layers before a sigmoid activated FC layer at the end, tested with different dropout rates.

### Instructions:

To run train & test using pre-trained **Resnet50**:

```bash
python train_test_resnet50.py -d [DIR_TO_DATA]
```

- ''-i', '--idx', type=int, required=False, default=1, help="Index number when saving graphs and model")
- '-e', '--epochs', type=int, required=False, default=10, help="Number of epochs for train & val")
- '-lr', '--learning_rate', type=float, required=False, default=1e-4, help="Learning rate for optimizer")
- '-b', '--batch', type=int, required=False, default=32, help="Number of batch size")
- '-d', '--data', type=str, required=False, default='/home/ubuntu/data/patchcamelyon', help="Dataset path")
- '-ts', '--train_size', type=int, required=False, default=2e18, help="Number of train dataset")
- '-vs', '--val_size', type=int, required=False, default=2e15, help="Number of train dataset")
- '-o', '--output', type=str, required=False, default='output', help="Directory for output")
- '-l', '--limit', type=bool, required=False, default=True, help="Limit GPU usage to avoid out of memory")



To run train & test using pre-trained **Densenet169**:

```bash
python train_test_densenet169.py
```



To test a trained model on test dataset:

```bash
python test.py -m [PATH_TO_MODEL] -d [DIR_TO_DATA]
```

- '-b', '--batch', type=int, required=False, default=32, help="Number of batch size")
- '-d', '--data', type=str, required=False, default='/home/ubuntu/data/patchcamelyon', help="Dataset path")
- '-m', '--model', type=str, required=False, default='model_10_epochs.h5', help="Dataset path")
- '-vs', '--val_size', type=int, required=False, default=2e15-1, help="Number of train dataset")
- '-o', '--output', type=str, required=False, default='output', help="Directory for output")
- '-l', '--limit', type=bool, required=False, default=True, help="Limit GPU usage to avoid out of memory")



To run **Tkinter GUI** script:

```python
python pcam_gui.py
```

To use **Tkinter GUI**:

* Select an image to analyze (Default images are provided in /sample_imgs folder)
* Select a model weight file (.h5) to use for prediction (Default folder is /model_ckpt)
* Press *Predict* to generate model prediction score

Note: Tkinter GUI may become temporarily slow/responsive during model prediction.

![alt text](D:\School\Course Material\Term 8\50.021 - Artificial Intelligence\Project\pcam_ai\sample_gui.PNG)





