**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Inventory Monitoring at Distribution Centers

Distribution centers often use robots to move objects as a part of their operations. Objects are carried in bins which can contain multiple objects. This project aims to develop an image processing model to detect the numbers of items in the bins in distribution centers. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items. It will further impove the automation of operations in distribution centers.
To build this project I used a pre-trained convolutional neural network (ResNet-50) to train the model with a subset of the Amazon Bin Image Dataset. I used SageMaker hyperparameter tuning tool to find the proper numbers of epoch and bach size, and learning rate for training prior to starting the major training job. The trained model is deployed as an endpoint on AWS platform, and can be consumed by sending requests to the endpoint. 


## Dataset

### Overview
This project used Amazon Bin Image Dataset, which contains 500,000 images of bins containing one or more objects. For each image there is a metadata file containing information about the image like the number of objects, it's dimension and the type of object. For this task, I only classify the number of objects in each bin. Therefore, I only used a subset of the image dataset and extracted the number of oobjects in each bin from the metadata. 

### Access
Use the code provided in`sagemaker.ipynb` to download and create dataset for model training. Upload the dataset to an existing or a newly created S3 bucket from the Jupyter notebook. Make sure that the role of the notebook instance has access to the S3 bucket. 

## Model Training
In this project I used a pre-trained convolutional neural network (ResNet-50) to train the model. The pre-trained ResNet-50 can classify images into object categories, and it can be used to classify new images by retain the neural network on a new calssification task. I used SageMaker Hyperparameter tunning job to search and refine the learning rate, batch size, and epoch. The script used for Hyperparameter tunning is documented in `hypo_tuner.py`.  Fine tuning these parameters is a good practice because it can significently increase the efficiency of the training job and potentially reduce the resource that is needed for the training. Then trained the model using SageMaker rescources and deployed the best model to an endpoint. The script used for the training is documented in `train.py`. 
Details of the training parameters and tranining results are documented in `sagemaker.ipynb` and `profiler-report.html`. 

## Machine Learning Pipeline
To deliever this project, the following phases are proceed:
1. Acquire data from an online source -> Extract useful metadata and Download data -> Upload data to S3 bucket
2. Feed subsets of the data in S3 bucket in Hyperparameter Tunning and Training jobs
3. Deploy best model to an AWS endpoint
4. Provide service by consuming requests 

## Tips for Saving Costs
To save costs for the development process and if you do not mind the training takes a bit longer, please consitder to train the model on a spot instance. Regarding increase the training efficiency, please consider to use the Hyperparameter Tunning job to search and finetune the training parameters before conducting the training. 