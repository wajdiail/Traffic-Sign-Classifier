# **Traffic Sign Recognition** 

## Introduction


The goal of this project is to classify German Traffic Signs.  Since this is a image recognition classification problem. A Deep learning Convolution network is employed to solve the task at hand.  This is because Convolution networks has been known to excel in image classification problems. Over the past years all the IMAGENET competition winners and top contenders are convolution network based models. The model used in this project is an modified version of LENET architecture from Yann Lecun's paper with some additional filters and strong regularization. The model was trained on a preprocessed data which involved data augmentation and normalization. The final validation and test accuracy obtained is 97% and 95.2%  respectively. 

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]:  ./ImagesPerClass.png "ImagesPerClass"
[image2]:  ./MoreStatistics.png "MoreStatistics"
[image3]:  ./10smallest.jpg "10smallest"
[image4]:  ./10largest.png "10largest"
[image5]:  ./10random.png "10random"
[image6]:  ./100random.png "100random"
[image7]:  ./ClassBar.png "ClassBar"
[image8]:  ./DAGImageBlur.jpg "DAGImageBLur"
[image9]:  ./DAGImageRot.jpg "DAGImageRot"
[image10]: ./BarAfterDag.png "BarAfterDag"
[image11]: ./100randomAfterPreprocess.png "100randomAfterPreprocess"
[image12]: ./DAGImageBlur.jpg "DAGImageBLur"
[image13]: ./DAGImageBlur.jpg "DAGImageBLur"


---

## Data Summary and Visualization:

Pandas library is used to calculate summary statistics of the traffic
signs data set:

#### Data set:

The set already given already included training, valid and test set. There was no need to segregate manually. 

No of classes: 43 <br />
No of images in the original training data set: 34799 <br />
No of images in the original validation data set: 4410 <br />
No of images in the original test data set: 12630 <br />

#### Image:

Image size: 32 X 32 X 3 <br />
Space : RGB <br />
Datatype: unit8 (0 to 255) <br />

##### Number of images per class:

![alt_text][image1]

#####  More statistics on the data set: 

![alt_text][image2]

##### Top 10 largest class by image count:

![alt_text][image4]

##### Top 10 smallest class by image count:

![alt_text][image3]

#### Exploratory visualization of the data set

##### 100 random samples from the entire data set

![alt_text][image6]

##### 10 random samples from 3 classes

![alt_text][image5]

##### Bar graph of the data set distribution with respect to classes

![alt_text][image7]

### Design and Test a Model Architecture

#### Preprocessing 

##### Data Augmentation:

During the exploratory visualization of the data set it is noted that the orignal data is unbalanced. Some classes have less images and some have more images comparatively. For example, in the dataset, label 0 and 19 have just 180 images whereas Label 2 and 1 has close to 2000 images. This is a problem, as it could lead to low precision and recall for certain classes. Data augmentation is a technique was used to generate more data by reasonably modifying the images. This also helps the model to generalize. There are lot of methods used for data augmentation. Some are image blurring, rotation, random cropping, panning, downscalling, flipping, inversing etc. We explored several technique and used some which helped solve this problem.  

The following the techniques are used in this model for generating the augmentated data.

Image Blur:

![alt_text][image8]

Image Rotation:

![alt_text][image9]

Other techniques such as image flipping, streching, downscaling (see commented section of the function _image_generator_ in the final code) were explored. But due to decrease in accuracy these techniques were not used in the final model. 

Stratergy for generating the images:

*Step1: Augmentated images were generated for all the classes of the data set by applying the above DAG techniques on every 10 images of the dataset.

Generate len(train_features)/10 images = 34799/10 = 3479x3(3 types of augmentation see image generator function for more details) = 10437 images 

In this step additional 10437 images were generated.
 
*Step2: During initial phase of training it is noted that the recall and precision has beeen low for the below class.
Also it is noted that these classes have less images compared to others. Hence additional images generated for these
classes. In this case 3 images were generated for every image of the selected class . For example: Class 20 has 299( 26249-25950) images. Hence it produced 299x3 images = 897 images

*Total number images in the trainig set after data augmentation: 52496*

*Note:* There is exists a predifined keras image generator but for learning purpose, a new images generator functon was written. However, the actualy image tranformation functions were used from scikit image library.


##### Bar graph after data augmentation:

![alt_text][image10]


##### Greyscale: 

In the paper “Traffic Sign Recognition with Multi-Scale Convolutional Networks”, Pierre Sermanet and Yann LeCun mentioned that using color channels did not seem to improve the classification accuracy. Hence RGB image is converted into greyscale images which decreases the number of features. 

##### Normalization: 

This is commonly used in most of the machine learning problem. It is to have zero mean and standard deviation which helps to penalize the losses more and reward the correct predictions less. 

##### Min Max scaling: 

This technique scales the feature values between 0 and 1. This also a requirement for adaptvie historgram equaliztion used as the last step of preprocessing pipeline. 

##### Adaptive Histogram Equalization: 

As a last step, adaptive histogram equalization is applied to increase the contrast of the greyscale images. This improved the exposure of darker images at same time not over exposuring other images. This method was also used in the paper... 

###### Visualziation of dataset after applying adaptive equalziation:

![alt text][image11]


#### Final model architecture: 


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Dropout		        | Keep Probability: 0.6 	                    |
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x12 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24	|
| RELU          		| 												|
| Dropout				| Keep Probability: 0.6 						|
| Max pooling 			| 2x2 stride, valid padding, outputs 5x5x12		|
| Fully Connected Layer	| inputs 600			     outputs 120		|
| RELU					|												|
| Dropout		        | Keep Probability: 0.6 	                    |
| Fully Connected Layer	| inputs 120			     outputs 84 		|
| RELU					|												|
| Fully Connected Layer	| inputs 84			 		 outputs 43 		|
 


#### Training Parameters

EPochs = 20 (After trail and error) <br /> 
Batch_size = 128 (Did not explored much here as this size did produce fairly good results) <br />
Optimizer: Adam (A default good optimizer)  <br />
Learning Rate: 0.001 <br />


#### Training Approch and Changes:

Finding solution to machline learning problems is always an iterative approch based on some trail and error. This solution is no different. Some key iterations and changes leading to the final model are discussed below.

##### Initial Run: 

The initial model has no dropout layers. The size of filters were half of the filters used in the final model. 
No data augmentation
RGB images were used directly.
Normalization 

Valid set accuracy : 88%

##### Change 1: 

No changes to initial model <br />
No data augmentation <br />
*Greyscale Images* <br />
*Normalization* <br />
*Min max scaling* <br />
*Adaptive Histogram equaliztion* <br />

Valid set accuracy: 92% <br />

###### Preicion and recall bar graph after running this model

##### Change 2:

No changes to initial model <br />
*Included data augmention* <br />
Greyscale Images <br />
Normalization <br />
Min max scaling <br />
Adaptive Histogram equaliztion <br />

Valid set accuracy: 95% <br />

##### Final Change: 

*Added dropout layer and double the filter sizes* <br />
*Included data augmention* <br />
Greyscale Images <br />
Normalization <br />
Min max scaling <br />
Adaptive Histogram equaliztion <br />


My final model results were: <br />
* validation set accuracy: 97% <br />
* test set accuracy of : 95% <br />

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


