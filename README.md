## Project Proposals



Team: SOTA Crush  
### Facebook Project: No 
### Project Title:  
Apply transfer learning and compare results between NFNET, EFFICIENT-NET on custom dataset COVID-19 chest Xray. 

 

### Project summary: 

Latest SOTA models NFNET, EFFICIENT-NET have been a very hot topics recently. We would like first to evaluate their performance on custom dataset COVID-19 chest Xray through transfer learning (as these models are very time consuming to train from scratch with Imagenet) and through the process we also want to deep dive into the new techniques that each network architecture offers for efficient training/inference.  

 

### Approach: 

Learn the architecture of NFNet and EFFICIENT-NET and use pretrained weights to evaluate the performance for each of them on the custom dataset. Try out different parameters to verify we achieve good results. 

Creating an ensemble models using these two networks to see if we get better results 

Scale down the networks using the same compound scaling method and compare the performance with baseline model like Resnet using knowledge distillation.  

### Datasets: 

- Small dataset: https://github.com/ieee8023/covid-chestxray-dataset 

- Large Dataset: https://www.kaggle.com/unaissait/curated-chest-xray-image-dataset-for-covid19

#### Resource: 

[1] High-Performance Large-Scale Image Recognition Without Normalization, Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan  

[2] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Minxing Tan, Quoc Le 

[3] Adversarial Examples Improve Image Recognition, Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan Yuille, Quoc V. Le 

[4] Pretrained Weights for model: https://github.com/rwightman/pytorch-image-models/#getting-started-documentation 

[5] Large dataset Covid-19 Xrays: https://www.kaggle.com/unaissait/curated-chest-xray-image-dataset-for-covid19 

[6] Small dataset Covid-19 Xrays:  https://github.com/ieee8023/covid-chestxray-dataset 

[7] Ensemble Tutorial: https://towardsdatascience.com/destroy-image-classification-by-ensemble-of-pre-trained-models-f287513b7687 
[8] Distilling the Knowledge in a Neural Network, G. Hinton et al 

 

### Team Members:  

- Bianzhu Fu 

- Yuqing Ren 

- Chuong Dao 

### TA Feed backs
Hi, I do have some questions after reading your proposal. You don't have to answer them immediately but might want to consider them while doing your project.

- 1. What do you mean by "Try out different parameters to verify we achieve good results". Is it model parameters(like different depth or width) or just hyper-parameters?
- 2. What evaluation criteria you would like to use? For some x-ray datasets, accuracy is not the standard evaluation criteria. They might be looking for something like AUC, etc. I don't know if it applies to your dataset but you will need to make it clear in your final report. 

Also, I really think you need to work on the "scale down part" as only doing the training and tuning part might be too trivial for the scope of the final project.


### FastAI tutorials and integrate with TIMM for transfer learning

https://docs.fast.ai/tutorial.vision.html

https://walkwithfastai.com/vision.external.timm

### Dataset Splits for the large dataset (3.5GB)
```
(60% training, 20% val, 20% test)

import splitfolders

splitfolders.ratio('dataset', output='output', seed=1024, ratio=(0.6,0.2,0.2))
```


