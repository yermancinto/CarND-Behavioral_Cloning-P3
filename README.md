# SDCND_Behavioral-Cloning

Project 3

My project includes the following files:

* BC15.py containing the script to create and train the model
* drive2.py for driving the car in autonomous mode
* model.h15 containing a trained convolution neural network
* This writeup summarizing the results


### Data acquisition

The quality of the data used to train the model will determine how well it predicts the steering angle:

* I trained the model using the mouse instead the keyboard, with soft and contnious angle changes.
* Vehicle was driven at around constant 9kph speed, as the simulator does.
* I got 4 complete laps on the main track. The data set us huge, takes hours training the model with all the data so in my last trials I realized than using 50% of these data is enough to have good results. 

* driving_log has 29258 lines, that means the size of the data set is 87774 images (29258x3)
* The size of train_set is 50% (43887 images)
* The size of test_set is 5% (4388 images)




### Design and Test a Model Architecture

#### 1. Augment the existing data Set:

My last dataset was big enough so I had no need to augment it. 
On my previous attemps I flipped the images and used x(-1) factor on the steering angle, but the results were not so good as I expected, so I dismissed this approach.

#### 2. Process the images:

Images are trimmed to focus the model on the road:

![trimmed_image](https://user-images.githubusercontent.com/41348711/45599921-5a1fea80-b9f4-11e8-84c6-6c63547f8bc8.JPG)

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 65x320x3 image| 
| Convolution2D     	| 24 filters , 5x5 kernel size, strides = (2,2), RELU activ.function 	|
| Dropout			|	25%											|
| Convolution2D     	| 36 filters , 5x5 kernel size, strides = (2,2), RELU activ.function 	|
| Dropout			|	25%											|
| Convolution2D     	| 48 filters , 5x5 kernel size, strides = (2,2), RELU activ.function 	|
| Dropout			|	25%											|
| Convolution2D     	| 64 filters , 3x3 kernel size, strides = (1,1), RELU activ.function 	|
| Dropout			|	25%											|
| Convolution2D     	| 64 filters , 3x3 kernel size, strides = (1,1), RELU activ.function 	|
| Dropout			|	25%											|
| Flatten		|										|
| Fully connected		|  output=100 , 5% - l2 regularizer    	|	
| Fully connected		|  output=50 , 5% - l2 regularizer    	|	
| Fully connected		|  output=10 , 5% - l2 regularizer    	|	
| Fully connected		|  output=1  	|	

* LOSS:  mean square error
* GENERATOR: Batch size=32 ; 50% train_size (43887 images); 5% test_size(4388 images)
* EPOCHS = 3


4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* Training set accuracy of 99.6%
* Validation set accuracy of 95.0%
* Test set accuracy of 92.0%

The proccess I followed was really trial and error 
First architecture choosen for the traffic sign classifier was the one from letnet, just modified to have (32,32,3) inputs images. No preprocces was applied to the images. learning rate was set to 0.001. The validation accuracy obtained with that model using 150 EPOCHS was lower than the needed 93%, so I started processing images before modifiying the model structure. 
* 1st) Convert images into grey scale and normalizing to have zero mean and equal standard deviation. Using 150 EPOCHS and 128 batch size got an accuracy of 90%
* 2nd) Use Histogram equalization tool in order to enhance low contrast images. Reducing the number of EPOCHS I was able to reach 91% accuracy
* 3rd) Add 0.5 droput after the 3rd layer. This change led me to get the needed 93% accuracy
* 4th) Using the recommendations from reviser I augmented the data set and change the equalization to a CLAHE histogram equalization. This way I got 95% accuracy for the validation data set

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Picture below shows six images randomly selected from the german traffic sign dataset and the prediction results using my final model: (http://benchmark.ini.rub.de/)

![imagen](https://user-images.githubusercontent.com/41348711/43659893-a69a4c2a-975d-11e8-8ea2-aee86bc0764b.png)

For this particular case I aws lucky and the result is 100% accuracy

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

Except for the second image, the NN identifies easily the traffic signs. It is possible that for the second image, the white spot beside the arrow confuses the NN giving a 28% probability to "No passing" traffic sign 

See below detailed predictions: 






