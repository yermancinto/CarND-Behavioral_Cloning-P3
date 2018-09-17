# SDCND_Behavioral Cloning

My project includes the following files:

* BC16.py containing the script to create and train the model
* drive2.py for driving the car in autonomous mode
* model.h16 containing a trained convolution neural network
* This writeup summarizing the results


### Data acquisition

The quality of the data used to train the model will determine how well it predicts the steering angle:

* I trained the model using the mouse instead the keyboard, with soft and contnious angle changes.
* Vehicle was driven at around constant 9kph speed, as the simulator does.
* I got 4 complete laps on the main track. The data set us huge, takes hours training the model with all the data so in my last trials I realized than using 50% of these data is enough to have good results. 

* driving_log has 29258 lines, that means the size of the data set is 87774 images (29258x3)
* The size of train_set is 10% (8777 images)
* The size of test_set is 5% (4388 images)


### Design and Test a Model Architecture

#### 1. Augment the existing data Set:

My last dataset was big enough so I had no need to augment it. 
On my previous attemps I flipped the images and used x(-1) factor on the steering angle, but the results were not so good as I expected, so I dismissed this approach.

#### 2. Process the images:

Images are trimmed to focus the model on the road:

![trimmed_image](https://user-images.githubusercontent.com/41348711/45599921-5a1fea80-b9f4-11e8-84c6-6c63547f8bc8.JPG)

Images are converted from BGR to RGB on the training proccess:

![read](https://user-images.githubusercontent.com/41348711/45601862-fdccc300-ba13-11e8-93a5-fdf5b5691930.JPG)

Left and right images are included in the training with a 0,3 correction factor (+0,3 for left images and -0,3 for the right ones) to help teach the car to return to the middle of the road.

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
* EPOCHS = 3 (higher epochs did not bring any improvement)

#### 4. Results

My final model results were:

* Training loss of  0.0221
* Test loss of 0.0207

![bc16_results](https://user-images.githubusercontent.com/41348711/45639797-3bcbf480-bab1-11e8-8018-582eacfe46ce.JPG)


First architecture I choosed for the project was LeNet. After tunning this model I was never able to drive autonomously beyond the stone bridge. 
Using the Nvidia model without any regularization technique results improved, the loss was reduced but still had overfitting problems. Furthermore the car was not able to run a complete lap autonmously.

25% Dropout was added to avoid overfitting after every 2D convolution layer. After that, % l2 regularizer was added for every fully connected layer.

Using the model.h16 model in the simulator the vehicle runs pretty good in all the sections of the track






