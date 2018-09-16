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

#### 3. Model architecture:

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

#### 4. Results

My final model results were:

* Training loss of  99.6%
* Test loss of 95.0%


First architecture I choosed for the project was the LeNet. After tunning this model I was never able to drive autonomously beyond the stone bridge. 
Using the Nvidia model without any regularization technique results improved, the loss was reduced but still had overfitting problems. Furthermore the car was not able to run a complete lap autonmously.
25% Dropout was added to avoid overfitting after every 2D convolution layer
5% l2 regularizer was added for every fully connected layer






