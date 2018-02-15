# **Traffic Sign Recognition** 

## Introduction


The goal of this project is to classify German Traffic Signs.  Since this is a image recognition classification problem. A Deep learning Convolution network is employed to solve the task at hand.  This is because CNNs are known to excel in image classification problems. Over the past years all the IMAGENET competition winners and top contenders are CNN based models. The model used in this project is an modified version of LeNeT architecture from Yann Lecun's paper with some additional filters and strong regularization. The model was trained on a preprocessed data which involved data augmentation and normalization. The final validation and test accuracy obtained is 97.3% and 95.6%  respectively. 

Here is a link to my [project code](https://github.com/wajdiail/Traffic-Sign-Classifier/blob/master/Final_Code.ipynb)
---

**Traffic Sign Recognition Project**

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
[image13]: ./web_images/img1_disp.jpeg
[image14]: ./web_images/img2_disp.jpeg
[image15]: ./web_images/img3_disp.jpeg
[image16]: ./web_images/img4_disp.jpeg
[image17]: ./web_images/img5_disp.jpeg
[image18]: ./WebSoftMax120.png
[image19]: ./WebSoftMaxRightofWay.png
[image20]: ./WebSoftMaxYield.png
[image21]: ./WebSoftMaxTurnLeft.png
[image22]: ./WebSoftMaxTurnRight.png
[image23]: ./FeatureMaps.png

---

## Data Summary and Visualization:

Pandas library is used for data summary and statistics of the traffic
signs data set:

#### Data set:

The data set given already includes training, valid and test set. There was no need them segregate manually. 

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

During the exploratory visualization of the data set it is noted that the orignal data is unbalanced. Some classes have less images and some have more images comparatively. For example, in the dataset, label 0 and 19 have only 180 images whereas Label 2 and 1 has close to 2000 images. This is a problem, as it could train the model to be biased on the classes which has more data which could lead to low precision and recall. Data augmentation is a technique used to generate more data by reasonably modifying the images. This also helps the model to generalize well. There are lot of methods used for data augmentation. Some are image blurring, rotation, random cropping, panning, downscalling, flipping, inversing etc. 

The following techniques are used in this model for generating the augmentated data.

**Image Blur:**

![alt_text][image8]

**Image Rotation:**

![alt_text][image9]

Other techniques such as image flipping, streching, downscaling (see commented section of the function _image_generator_ in the final code) were explored. But due to decrease in accuracy these techniques were not used in the final model. 

**Steps involved in generating the images**:

* Step1: Augmentated images were generated for all the classes of the data set by applying the above techniques on every 10 images of the dataset. Generated len(train_features)/10 images = 34799/10 = 3479x3(3 types of augmentation see image generator function for more details) = 10437 images 

 **In this step additional 10437 images were generated**
 
* Step2: During the initial phase of training it is noted that the recall and precision has beeen low for some classes.
Also it is noted that these classes have less images compared to others. Hence additional images were generated for these
classes. In this step three images were generated for every single image of the selected class. For example: Class 20 has 299( 26249-25950) images. Hence it produced 299x3 images = 897 images

**Total number images in the trainig set after data augmentation: 52496**

**Note:** There exists a predifined keras image generator but for learning purpose, a new image generator functon was written. However, the actualy image tranformation functions were used from scikit image library.


### Bar graph after data augmentation:

![alt_text][image10]


##### Greyscale: 

In the paper “Traffic Sign Recognition with Multi-Scale Convolutional Networks”, Pierre Sermanet and Yann LeCun mentioned that using color channels did not seem to improve the classification accuracy. Hence RGB images were converted into greyscale images which also decreases the number of features. 

##### Normalization: 

Normalization is commonly used in most of the machine learning problem. It is to have zero mean and standard deviation which helps to penalize the losses more and reward the correct predictions less. This also aids gradient descent.

##### Min Max scaling: 

Min-max technique scales the feature values between 0 and 1. This is also a requirement for adaptvie historgram equaliztion used as the last step of preprocessing pipeline. 

##### Adaptive Histogram Equalization: 

In the last step, adaptive histogram equalization is applied to increase the contrast of the greyscale images. This improves the exposure of darker images at same time not over exposuring other images. This method is also used in the paper “Traffic Sign Recognition with Multi-Scale Convolutional Networks”, by Pierre Sermanet and Yann LeCun

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

Finding solution to machine learning problems is always an iterative approch based on some trail and error. This solution is no different. Some key iterations and changes leading to the final model are discussed below.

##### Initial Run: 

The initial model had no dropout layers. The size of filters were half of the filters used in the final model <br />
No data augmentation applied <br />
RGB images were used directly <br />
Only Normalization was used in preprocessing <br />

Valid set accuracy : 88%

##### Change 1: 

RGB images were converted to Greyscale images <br />
Min max scaling was applied after normalization <br />
Improved the contrast using Adaptive Histogram Equaliztion <br />
Generated precision and recall bar graph for fine tuning <br />

Valid set accuracy: 92% <br />

##### Change 2:

Generated more images in general for the entire training set and additional images for classes having low recall and precison values using data augmention techniques<br />

Valid set accuracy: 95% <br />

##### Final Change: 

Added dropout layer and doubled the filter sizes <br />


Final model results were: <br />
* validation set accuracy: **97.3%** <br />
* test set accuracy of : **95.6%** <br />
* web images accuracy : **80%** <br />

 

### Testing Model on Web Images

Here are five German traffic signs that are found on the web:

![alt text][image13] ![alt text][image14] ![alt text][image15] 
![alt text][image16] ![alt text][image17]


Here are the results of the prediction:

| Image				  	  	           |     Prediction	        					| 
|:------------------------------------:|:------------------------------------------:| 
| Speed limit 120(km/h)	               | Speed limit 60(km/h)  						| 
| Right of way at the next intersection| Right of way at the next intersection 		|
| Turn left a head					   | Turn left a head							|
| Yield	      		                   | Yield  					 				|
| Turn right a head			           | Turn right a head							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of **80%**


#### Top 5 Softmax visualization of the web images:

For the first image, 120km/h, the model is incorrectly predicting 60 km/h. Initial analysis did not flag anything unsual. Both the classes 60km/h and 120km/h have equal amount of data and relatively good percentage of data in the overall training set. Even the precision and recall numbers are high for the both classes. Hence it could be suspected that the model should be underfitting for this class.   

![alt text][image18]

The model has correctly predicted this image. However, the confidence is slighly lesser. 

![alt text][image19]

The model has predicted correctly with good confidence

![alt text][image20]

In this case, though the model has correctly predicted the images but its confidence is less and it is getting confused with its opposite side image. 

![alt text][image21]

Similarly, for this image the close predictions are the ones which has similar shape.

![alt text][image22]


#### Feature map from the model:

Feauter maps are extracted from the model to have a better understanding of the internal functionality of ConvNets. Below is the visualization of two activation maps.

![alt text][image23]


### Future Improvements:

1. Precision/Recall are less for few classes. This could be improved by better targetting the data augmentation
2. The model is generally weak in predicting traffic signs with numbers. Further analysis should be done here to improve.
3. Multi-Scale CNNs can be employed to improve the accuracy further
4. Keras' built in data generator can be used for augmenting the data.
5. Train for more Epochs on GPU









