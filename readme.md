
# What is the Task?
* The goal is to find an efficient approach using a single monocular camera to perform real-time traffic signs recognition and distance estimation between traffic sign and viewpoint.
The next thing we want to apply, after having the distance, would be a simple feedback system for the driver as a guideline. One thing to consider is that we do not want to distract the driver by giving too much feedback, therefore, we will take the most important traffic signs which are a kind of safety feedback or an alert for the driver such as getting closer to the pedestrian path.

* ![classes](result.gif) <br />
# Demo
* Watch Demo on  youtube: https://www.youtube.com/watch?v=JZ-7B4QJj7o


## Prerequisites:
- Python 3.5
- [OpenCV3](https://opencv.org/)
- tensorflow
- keras
- scikit-image
# Motivation
* Traffic signs are an integral component of road transport infrastructure. They deliver essential driving information to road users, which in turn require them to adapt their driving behavior to road regulations.
* Traffic sign recognition (TSR) is a technology by which a vehicle is able to recognize the traffic signs put on the road e.g. "Give Way" or "Pedestrian Path".
# Dataset
We have used the **DFG traffic sign dataset** consisting of 200 categories including a large number of traffic signs with high intra-category appearance variations.[link to the dataset](https://www.vicos.si/resources/dfg/)<br>
More precisely, Our dataset consists of 200 traffic sign categories captured in Slovenian roads spanning around 7000 high-resolution images. Images were provided and annotated by a Slovenian company DFG Consulting d.o.o.
Additional augmentation images were created by inserting cropped instances of traffic signs into the Belgium Traffic Sign Dataset.
<br> **Note:** We add a human readable annotation as well.<br>
The RGB images were acquired with a camera mounted on a vehicle that was driven through six different Slovenian municipalities. The image data was acquired in rural as well as urban areas. Only images containing at least one traffic sign were selected from the vast corpus of collected data. Moreover, the selection was performed in such a way that there is usually a significant scene change between any pair of selected consecutive images.
you can see some of the classes in the following image:

![classes](/images/classes.png)


# Evaluation
The evaluation dataset termed DFG Traffic Sign Dataset was created by focusing only on planar traffic signs with a sufficient number of samples available. Each category has at least 20 instances. Samples with a bounding box size of at least 30 pixels are tightly annotated.

# Methods
In this project, we have presented a supervised method for **recognizing trafﬁc signs** based on deep Convolutional Neural Networks (CNNs). We have used **MaskRCNN** and **Google Colaboratory**, a free environment that runs entirely in the cloud and provides a GPU.<br>
For estimating **distance** the **bird's-eye view** method is taken to account.<br>
we have tried the feedback system to become simple as possible. Thus, we have only focused on the core approach of our simulation which depends on two main things:
* knowing the distance to the traffic sign
* provide an appropriate command depending on the distance and type of the traffic sign

<br>
Later on, we will try to implement a full simulation environment as future work.

# Note #
* We faced some errors due to using the latest versions of TensorFlow, Keras, etc. So, we decided to install the older versions to get rid of the errors.
```sh
!pip install scikit-image==0.16.2
!pip install h5py==2.10.0
!pip install tensorflow==1.15
!pip install tensorflow-gpu==1.15.0
!pip install keras==2.1.6
```

# Some highlights
* 83% accuracy over the test set
* Learning rate annealing, Dropout increment, batch size increase as accuracy increases
* Greedy best save implemented on validation accuracy being the criteria

# Model Training
In order to use Mask_RCNN, some configurations must be set based on the desired task.

* We have **200** classes which are related to different trafic signs such as stop, give way, keep right and so on.<br>
So, the overall number of classes is equal to **201**: **Background + our 200 classes**

* Additionally, required libraries from **matterport implementation** of Mask_RCNN are used.

```sh
!git clone https://github.com/matterport/Mask_RCNN.git
```
* Use of complete augmented dataset where some of images are changed using augmentation techniques, maximum sample target over 5 epochs.
<br/>


We need to create a synthetic dataset
Extend the Dataset class and add a method to load the traffic signs dataset. Something to pay attention is that, since we want to train our custom model using matterport implementation, we need to override some functions such as:

* load_image()
* load_mask()
* image_reference()

We have used Mask_RCNN to train the model.
Since we are in the training phase, **mode is set to 'training'**.
Number of epochs is set to 5.
<br>

Train can be done in two stages:

- Only the `heads`: Here we're freezing all the backbone layers and training only the randomly initialized layers. To train only the head layers, pass layers="heads" to the train() function.

- Fine-tune `all` layers: Here we're activating all the backbone layers and training all of them. To train all layers simply pass layers="all" to train all layers.
We will retrain all layers of the model.


# config
The `Config` class contains the default configuration of the Matterport implementation. We need to subclass it and modify the attributes that must be changed.

We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the CocoConfig class in coco.py from the repository.
Matterport guys have been providing pre-trained weights for MS COCO to make it easier to start. these weights can be used as a starting point to train our own variation on the network. <br>
We have used the weights of the pre-trained **mask_rcnn_coco** model to initialize the weights for training the model.

# Model Testing

The model is tested with a variety of images after the test set is evaluated.
The ***goal*** of this part is to **predict** with the **'Inference' Mode**.
The best model founded on the training phase (mask_rcnn_road_cfg_0005)is used in the test phase.
To evaluate the model, despite the visual representation that we have shown so far, we are going to compute the  `Average Precision` at a set `IoU` threshold. The default value for the `Intersection over Union (IoU)` threshold is 0.5.

There are some other metrics as follows:
- precisions: List of precisions at different class score thresholds.
- recalls: List of recall values at different class score thresholds.
- overlaps: [pred_boxes, gt_boxes] IoU overlaps.

We are going to consider only the Mean Average Precision metric in our evaluation function.


# Distance estimation and Feedback

we’ll try to assess the distance to the detected traffic signs from the viewpoint.

Once we get the bounding box and midpoint of its bottom edge, using perspective transform, we can map the original image to the `bird's view`.

We do not need to warp the whole image, but just recalculate the position of the single on the warped image. On that image, there is a direct correlation between the pixel position and distance in meters, so the distance between the calculated position of the midpoint and the bottom of the image multiplied by the number of meters per pixel represents the distance between our viewpoint and the traffic sign that we have detected.

**Later on once we have the distance per frame,
by looking at how that distance changes, we will have a feedback message to the driver as a guideline during the driving.**
## Bird's eye view
As mentioned above, we are going to take a top view of our image which is known as a bird's eye view in the literature.
To provide this view we need to define the homography function.
In the field of computer vision, any two images of the same planar surface in space are related by homography (assuming a pinhole camera model). This has many practical applications, such as image rectification, image registration, or camera motion—rotation and translation—between two images. Briefly, the planar homography relates to the transformation between two planes (up to a scale factor).

![classes](/images/bird.png)

# Results
Let's have everything in one place!
![classes](/images/result.png)
![classes](/images/download.png)

# Future developments
- Better performance with higher framerate
- Use other approaches like YOLO or SSD and Detectron implementation
- Dynamic image processing
- No need to retrain the model when running the program
- provide the TTR to prevent driver distraction by giving too much feedback
