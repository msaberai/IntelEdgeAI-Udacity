# Lesson 2 - Leveraging Pre-Trained Models

In this lesson the following topics will be covered:

- Basics of the Intel® Distribution OpenVINO™ Toolkit

- Different Computer Vision model types

- Available Pre-Trained Models in the Software

- Choosing the right Pre-Trained Model for your App

- Loading and Deploying a Basic App with a Pre-Trained Model

## The Intel® Distribution of OpenVINO™ Toolkit

- The OpenVINO™ Toolkit’s name comes from “<b>Open</b> <b>V</b>isual <b>I</b>nferencing and Neural <b>N</b>etwork <b>O</b>ptimization”. It is largely focused around optimizing neural network inference, and is open source.

- It is developed by Intel®, and helps support fast inference across Intel® CPUs, GPUs, FPGAs and Neural Compute Stick with a common API. OpenVINO™ can take models built with multiple different frameworks, like TensorFlow or Caffe, and use its Model Optimizer to optimize for inference. This optimized model can then be used with the Inference Engine, which helps speed inference on the related hardware. It also has a wide variety of Pre-Trained Models already put through Model Optimizer.

- By optimizing for model speed and size, OpenVINO™ enables running at the edge. This does not mean an increase in inference accuracy - this needs to be done in training beforehand. The smaller, quicker models OpenVINO™ generates, along with the hardware optimizations it provides, are great for lower resource applications. For example, an IoT device does not have the benefit of multiple GPUs and unlimited memory space to run its apps.

## Pre-Trained Models in OpenVINO™

In general, pre-trained models refer to models where training has already occurred, and often have high, or even cutting-edge accuracy. Using pre-trained models avoids the need for large-scale data collection and long, costly training. Given knowledge of how to preprocess the inputs and handle the outputs of the network, you can plug these directly into your own app.

In OpenVINO™, Pre-Trained Models refer specifically to the Model Zoo, in which the Free Model Set contains pre-trained models already converted using the Model Optimizer. These models can be used directly with the Inference Engine.

## Types of Computer Vision Models

We covered three types of computer vision models in the video: Classification, Detection, and Segmentation.

Classification determines a given “class” that an image, or an object in an image, belongs to, from a simple yes/no to thousands of classes. These usually have some sort of “probability” by class, so that the highest probability is the determined class, but you can also see the top 5 predictions as well.

Detection gets into determining that objects appear at different places in an image, and oftentimes draws bounding boxes around the detected objects. It also usually has some form of classification that determines the class of an object in a given bounding box. The bounding boxes have a confidence threshold so you can throw out low-confidence detections.

Segmentation classifies sections of an image by classifying each and every pixel. These networks are often post-processed in some way to avoid phantom classes here and there. Within segmentation are the subsets of semantic segmentation and instance segmentation - the first wherein all instances of a class are considered as one, while the second actually consider separates instances of a class as separate objects.

## Case Studies in Computer Vision

We focused on SSD, ResNet and MobileNet in the video. SSD is an object detection network that combined classification with object detection through the use of default bounding boxes at different network levels. ResNet utilized residual layers to “skip” over sections of layers, helping to avoid the vanishing gradient problem with very deep neural networks. MobileNet utilized layers like 1x1 convolutions to help cut down on computational complexity and network size, leading to fast inference without substantial decrease in accuracy.

One additional note here on the ResNet architecture - the paper itself actually theorizes that very deep neural networks have convergence issues due to exponentially lower convergence rates, as opposed to just the vanishing gradient problem. The vanishing gradient problem is also thought to be helped by the use of normalization of inputs to each different layer, which is not specific to ResNet. The ResNet architecture itself, at multiple different numbers of layers, was shown to converge faster during training than a “plain” network without the residual layers.

## Available Pre-Trained Models in OpenVINO™

Most of the Pre-Trained Models supplied by OpenVINO™ fall into either face detection, human detection, or vehicle-related detection. There is also a model around detecting text, and more!

Models in the Public Model Set must still be run through the Model Optimizer, but have their original models available for further training and fine-tuning. The Free Model Set are already converted to Intermediate Representation format, and do not have the original model available. These can be easily obtained with the Model Downloader tool provided in the files installed with OpenVINO™.

The SSD and MobileNet architectures we discussed previously are often the main part of the architecture used for many of these models.

## Solution: Loading Pre-Trained Models

### Choosing Models

The following models were chosen for the three tasks:

   - Human Pose Estimation: `human-pose-estimation-0001`
   - Text Detection: `text-detection-0004`
   - Determining Car Type & Color: `vehicle-attributes-recognition-barrier-0039`

### Downloading Models
To navigate to the directory containing the Model Downloader:
    `cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader`

Within there, you'll notice a `downloader.py` file, and can use the `-h` argument with it to see available arguments. For this exercise, `--name` for model name, and `--precisions`, used when only certain precisions are desired, are the important arguments. Note that running `downloader.py` without these will download all available pre-trained models, which will be multiple gigabytes. You can do this on your local machine, if desired, but the workspace will not allow you to store that much information.

Note: In the classroom workspace, you will not be able to write to the `/opt/intel` directory, so you should also use the `-o` argument to specify your output directory as `/home/workspace` (which will download into a created intel folder therein).

Downloading Human Pose Model

``` sudo ./downloader.py --name human-pose-estimation-0001 -o /home/workspace ```

Downloading Text Detection Model

```sudo ./downloader.py --name text-detection-0004 --precisions FP16 -o /home/workspace ```

Downloading Car Metadata Model

``` sudo ./downloader.py --name vehicle-attributes-recognition-barrier-0039 --precisions INT8 -o /home/workspace ```

### Verifying Downloads

The downloader itself will tell you the directories these get saved into, but to verify yourself, first start in the `/home/workspace` directory (or the same directory as the Model Downloader if on your local machine without the `-o` argument). From there, you can `cd intel`, and then you should see three directories - one for each downloaded model. Within those directories, there should be separate subdirectories for the precisions that were downloaded, and then .xml and .bin files within those subdirectories, that make up the model.

## Optimizations on the Pre-Trained Models

In the exercise, you dealt with different precisions of the different models. Precisions are related to floating point values - less precision means less memory used by the model, and less compute resources. However, there are some trade-offs with accuracy when using lower precision. There is also fusion, where multiple layers can be fused into a single operation. These are achieved through the Model Optimizer in OpenVINO™, although the Pre-Trained Models have already been run through that process. We’ll return to these optimization techniques in the next lesson.

## Choosing the Right Model for Your App

Make sure to test out different models for your application, comparing and contrasting their use cases and performance for your desired task. Remember that a little bit of extra processing may yield even better results, but needs to be implemented efficiently.

This goes both ways - you should try out different models for a single use case, but you should also consider how a given model can be applied to multiple use cases. For example, being able to track human poses could help in physical therapy applications to assess and track progress of limb movement range over the course of treatment.

## Pre-processing Inputs

The pre-processing needed for a network will vary, but usually this is something you can check out in any related documentation, including in the OpenVINO™ Toolkit documentation. It can even matter what library you use to load an image or frame - OpenCV, which we’ll use to read and handle images in this course, reads them in the `BGR` format, which may not match the `RGB` images some networks may have used to train with.

Outside of channel order, you also need to consider image size, and the order of the image data, such as whether the color channels come first or last in the dimensions. Certain models may require a certain normalization of the images for input, such as pixel values between 0 and 1, although some networks also do this as their first layer.

In OpenCV, you can use `cv2.imread` to read in images in `BGR` format, and `cv2.resize` to resize them. The images will be similar to a numpy array, so you can also use array functions like `.transpose` and `.reshape` on them as well, which are useful for switching the array dimension order.

## Solution: Pre-processing Inputs

Using the documentation pages for each model, I ended up noticing they needed essentially the same preprocessing, outside of the height and width of the input to the network. The images coming from `cv2.imread` were already going to be BGR, and all the models wanted BGR inputs, so I didn't need to do anything there. However, each image was coming in as height x width x channels, and each of these networks wanted channels first, along with an extra dimension at the start for batch size.

So, for each network, the preprocessing needed to 1) re-size the image, 2) move the channels from last to first, and 3) add an extra dimension of `1` to the start. Here is the function I created for this, which I could call for each separate network:

``` python
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
```
Then, for each model, just call this function with the height and width from the documentation:

Human Pose
``` python
preprocessed_image = preprocessing(preprocessed_image, 256, 456)
```

Text Detection
``` python
preprocessed_image = preprocessing(preprocessed_image, 768, 1280)
```

Car Meta
``` python
preprocessed_image = preprocessing(preprocessed_image, 72, 72)
```

Testing
To test your implementation, you can just run `python test.py`.

## Handling Network Outputs

Like the computer vision model types we discussed earlier, we covered the primary outputs those networks create: classes, bounding boxes, and semantic labels.

Classification networks typically output an array with the softmax probabilities by class; the argmax of those probabilities can be matched up to an array by class for the prediction.

Bounding boxes typically come out with multiple bounding box detections per image, which each box first having a class and confidence. Low confidence detections can be ignored. From there, there are also an additional four values, two of which are an X, Y pair, while the other may be the opposite corner pair of the bounding box, or otherwise a height and width.

Semantic labels give the class for each pixel. Sometimes, these are flattened in the output, or a different size than the original image, and need to be reshaped or resized to map directly back to the input. 

## Running Your First Edge App

You have now learned the key parts of working with a pre-trained model: obtaining the model, preprocessing inputs for it, and handling its output. In the upcoming exercise, you’ll load a pre-trained model into the Inference Engine, as well as call for functions to preprocess and handle the output in the appropriate locations, from within an edge app. We’ll still be abstracting away some of the steps of dealing with the Inference Engine API until a later lesson, but these should work similarly across different models.

## Solution: Deploy an App at the Edge

This was a tough one! It takes a little bit to step through this solution, as I want to give you some of my own techniques to approach this rather difficult problem first. The solution video is split into three parts - the first focuses on adding in the preprocessing and output handling calls within the app itself, and then into how I would approach implementing the Car Meta model's output handling.

### Early Steps and Car Meta Model Output Handling

The code for calling preprocessing and utilizing the output handling functions from within `app.py` is fairly straightforward:

``` python
preprocessed_image = preprocessing(image, h, w)
```

This is just feeding in the input image, along with height and width of the network, which the given inference_network.load_model function actually returned for you.

``` python
output_func = handle_output(args.t)
processed_output = output_func(output, image.shape)
```

This is partly based on the helper function I gave you, which can return the correct output handling function by feeding in the model type. The second line actually sends the output of inference and image shape to whichever output handling function is appropriate.

### Car Meta Output Handling

Given that the two outputs for the Car Meta Model are `"type"` and `"color"`, and are just the softmax probabilities by class, I wanted you to just return the `np.argmax`, or the index where the highest probability was determined.

``` python
def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # Get rid of unnecessary dimensions
    color = output['color'].flatten()
    car_type = output['type'].flatten()
    # TODO 1: Get the argmax of the "color" output
    color_pred = np.argmax(color)
    # TODO 2: Get the argmax of the "type" output
    type_pred = np.argmax(car_type)

    return color_pred, type_pred
 ```

#### Run the Car Meta Model

I have moved the models used in the exercise into a models subdirectory in the `/home/workspace directory`, so the path used can be a little bit shorter.

`python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"`

For the other models, make sure to update the input image `-i`, model type `-t`, and model `-m` accordingly.

### Pose Estimation Output Handling

Handling the car output was fairly straightforward by using `np.argmax`, but the outputs for the pose estimation and text detection models is a bit trickier. However, there's a lot of similar code between the two. In this second part of the solution, I'll go into detail on the pose estimation model, and then we'll finish with a quick video on handling the output of the text detection model.

Pose Estimation is more difficult, and doesn't have as nicely named outputs. I noted you just need the second one in this exercise, called `'Mconv7_stage2_L2'`, which is just the keypoint heatmaps, and not the associations between these keypoints. From there, I created an empty array to hold the output heatmaps once they are re-sized, as I decided to iterate through each heatmap 1 by 1 and re-size it, which can't be done in place on the original output.

```` python
def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    heatmaps = output['Mconv7_stage2_L2']
    # TODO 2: Resize the heatmap back to the size of the input
    # Create an empty array to handle the output map
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    # Iterate through and re-size each heatmap
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap
````

Note that the `input_shape[0:2][::-1]` line is taking the original image shape of HxWxC, taking just the first two (HxW), and reversing them to be WxH as `cv2.resize` uses.

### Text Detection Model Handling

Text Detection had a very similar output processing function, just using the `'model/segm_logits/add'` output and only needing to resize over two "channels" of output. I likely could have extracted this out into its own output handling function that both Pose Estimation and Text Detection could have used.

``` python
def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    text_classes = output['model/segm_logits/add']
    # TODO 2: Resize this output back to the size of the input
    out_text = np.empty([text_classes.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(text_classes[0])):
        out_text[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])

    return out_text
 ```

## Lesson Glossary

### Edge Application

Applications with inference run on local hardware, sometimes without network connections, such as Internet of Things (IoT) devices, as opposed to the cloud. Less data needs to be streamed over a network connection, and real-time decisions can be made.

### OpenVINO™ Toolkit

The Intel® Distribution of OpenVINO™ Toolkit enables deep learning inference at the edge by including both neural network optimizations for inference as well as hardware-based optimizations for Intel® hardware.

### Pre-Trained Model

Computer Vision and/or AI models that are already trained on large datasets and available for use in your own applications. These models are often trained on datasets like ImageNet. Pre-trained models can either be used as is or used in transfer learning to further fine-tune a model. The OpenVINO™ Toolkit provides a number of pre-trained models that are already optimized for inference.

### Transfer Learning

The use of a pre-trained model as a basis for further training of a neural network. Using a pre-trained model can help speed up training as the early layers of the network have feature extractors that work in a wide variety of applications, and often only late layers will need further fine-tuning for your own dataset. OpenVINO™ does not deal with transfer learning, as all training should occur prior to using the Model Optimizer.

### Image Classification

A form of inference in which an object in an image is determined to be of a particular class, such as a cat vs. a dog.

### Object Detection

A form of inference in which objects within an image are detected, and a bounding box is output based on where in the image the object was detected. Usually, this is combined with some form of classification to also output which class the detected object belongs to.

### Semantic Segmentation

A form of inference in which objects within an image are detected and classified on a pixel-by-pixel basis, with all objects of a given class given the same label.

### Instance Segmentation

Similar to semantic segmentation, this form of inference is done on a pixel-by-pixel basis, but different objects of the same class are separately identified.

### SSD

A neural network combining object detection and classification, with different feature extraction layers directly feeding to the detection layer, using default bounding box sizes and shapes/

### YOLO

One of the original neural networks to only take a single look at an input image, whereas earlier networks ran a classifier multiple times across a single image at different locations and scales.

### Faster R-CNN

A network, expanding on R-CNN and Fast R-CNN, that integrates advances made in the earlier models by adding a Region Proposal Network on top of the Fast R-CNN model for an integrated object detection model.

### MobileNet

A neural network architecture optimized for speed and size with minimal loss of inference accuracy through the use of techniques like 1x1 convolutions. As such, MobileNet is more useful in mobile applications that substantially larger and slower networks.

### ResNet

A very deep neural network that made use of residual, or “skip” layers that pass information forward by a couple of layers. This helped deal with the vanishing gradient problem experienced by deeper neural networks.

### Inception

A neural network making use of multiple different convolutions at each “layer” of the network, such as 1x1, 3x3 and 5x5 convolutions. The top architecture from the original paper is also known as GoogLeNet, an homage to LeNet, an early neural network used for character recognition.

### Inference Precision

Precision refers to the level of detail to weights and biases in a neural network, whether in floating point precision or integer precision. Lower precision leads to lower accuracy, but with a positive trade-off for network speed and size.





