# Intel® Edge AI Scholarship Program
Build and deploy AI models at the edge! Leverage the potential of edge computing using the Intel® Distribution of OpenVINO™ toolkit to fast-track development of high-performance computer vision and deep learning inference applications. Apply to the Intel® Edge AI Fundamentals Course for the chance to participate in a vibrant student community and to earn one of 750 Nanodegree program scholarships to continue your education with the Intel Distribution of OpenVINO toolkit and other computer vision tools.

## How it works
This scholarship is open to all applicants interested in learning how to work with computer vision deep learning models at the edge. Applicants 18 years of age or older are invited to apply.

We’ll review all applications and select recipients to participate in the Intel® Edge AI Fundamentals Course. This is where your learning begins! Recipients will spend 2.5 months optimizing powerful computer vision deep learning models. The initial fundamentals course is intended for students with some background in AI and computer vision, with experience in either Python or C++ and familiarity with command line basics.

Top students from the initial fundamentals course will be selected for one of 750 follow-up scholarships to the brand-new Intel® Edge AI for IoT Developers Nanodegree program.

-----------------------------------------------------------------------------------------------------------

# Lesson 1 - Introduction to AI at the Edge
This lesson first takes a brief look at AI at the Edge, its importance, different edge applications, and some of the history behind it. 


## What is AI at the Edge?

- The edge means local (or near local) processing, as opposed to just anywhere in the cloud. This can be an actual local device like a smart refrigerator, or servers located as close as possible to the source (i.e. servers located in a nearby area instead of on the other side of the world).

- The edge can be used where low latency is necessary, or where the network itself may not always be available. The use of it can come from a desire for real-time decision-making in certain applications.

- Many applications with the cloud get data locally, send the data to the cloud, process it, and send it back. The edge means there’s no need to send to the cloud; it can often be more secure (depending on edge device security) and have less impact on a network. Edge AI algorithms can still be trained in the cloud, but get run at the edge.

## Why is AI at the Edge Important?

- Network communication can be expensive (bandwidth, power consumption, etc.) and sometimes impossible (think remote locations or during natural disasters)

- Real-time processing is necessary for applications, like self-driving cars, that can't handle latency in making important decisions

- Edge applications could be using personal data (like health data) that could be sensitive if sent to cloud

- Optimization software, especially made for specific hardware, can help achieve great efficiency with edge AI models


## Applications of AI at the Edge

- There are nearly endless possibilities with the edge.

- IoT devices are a big use of the edge.

- Not every single app needs it - you can likely wait a second while your voice app goes to ask the server a question, or such as when NASA engineers are processing the latest black hole data.

## Historical Context

- Cloud computing has gotten a lot of the news in recent years, but the edge is also growing in importance.

- Per Intel®, IoT growth has gone from 2 billion devices in 2006 to a projected 200 billion by 2020.

- From the first network ATMs in the 1970's, to the World Wide Web in the 90's, and on up to smart meters in early 2000's, we've come a long way.

- From the constant use devices like phones to smart speakers, smart refrigerators, locks, warehouse applications and more, the IoT pool keeps expanding.

## Why Are the Topics Distinct?

 - Pre-trained models can be used to explore your options without the need to train a model. This pre-trained model can then be used with the Inference Engine, as it will already be in IR format. This can be integrated into your app and deployed at the edge.

- If you created your own model, or are leveraging a model not already in IR format (TensorFlow, PyTorch, Caffe, MXNet, etc), use the Model Optimizer first. This will then feed to the Inference Engine, which can be integrated into your app and deployed at the edge.

- While you'll be able to perform some amazingly efficient inference after feeding into the Inference Engine, you'll still want to appropriately handle the output for the edge application, and that's what we'll hit in the final lesson.

-----------------------------------------------------------------------------------------------------------

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

-----------------------------------------------------------------------------------------------------------

# Lesson 3 - The Model Optimizer

In this lesson we'll cover:

   - Basics of the Model Optimizer
    
   - Different Optimization Techniques and their impact on model performance
    
   - Supported Frameworks in the Intel® Distribution of OpenVINO™ Toolkit
    
   - Converting from models in those frameworks to Intermediate Representations
    
   - And a bit on Custom Layers

## The Model Optimizer

The Model Optimizer helps convert models in multiple different frameworks to an Intermediate Representation, which is used with the Inference Engine. If a model is not one of the pre-converted models in the Pre-Trained Models OpenVINO™ provides, it is a required step to move onto the Inference Engine.

As part of the process, it can perform various optimizations that can help shrink the model size and help make it faster, although this will not give the model higher inference accuracy. In fact, there will be some loss of accuracy as a result of potential changes like lower precision. However, these losses in accuracy are minimized. 

### Local Configuration

Configuring the Model Optimizer is pretty straight forward for your local machine, given that you already have OpenVINO™ installed. You can navigate to your OpenVINO™ install directory first, which is usually `/opt/intel/openvino`. Then, head to `/deployment_tools/model_optimizer/install_prerequisites`, and run the `install_prerequisites.sh` script therein.

## Optimization Techniques

Here,the focus is ainly on three optimization techniques: quantization, freezing and fusion.

### Quantization

Quantization is related to the topic of precision I mentioned before, or how many bits are used to represent the weights and biases of the model. During training, having these very accurate numbers can be helpful, but it’s often the case in inference that the precision can be reduced without substantial loss of accuracy. Quantization is the process of reducing the precision of a model.

With the OpenVINO™ Toolkit, models usually default to FP32, or 32-bit floating point values, while FP16 and INT8, for 16-bit floating point and 8-bit integer values, are also available (INT8 is only currently available in the Pre-Trained Models; the Model Optimizer does not currently support that level of precision). FP16 and INT8 will lose some accuracy, but the model will be smaller in memory and compute times faster. Therefore, quantization is a common method used for running models at the edge.

### Freezing

Freezing in this context is used for TensorFlow models. Freezing TensorFlow models will remove certain operations and metadata only needed for training, such as those related to backpropagation. Freezing a TensorFlow model is usually a good idea whether before performing direct inference or converting with the Model Optimizer.

### Fusion

Fusion relates to combining multiple layer operations into a single operation. For example, a batch normalization layer, activation layer, and convolutional layer could be combined into a single operation. This can be particularly useful for GPU inference, where the separate operations may occur on separate GPU kernels, while a fused operation occurs on one kernel, thereby incurring less overhead in switching from one kernel to the next.

## Supported Frameworks

The supported frameworks with the OpenVINO™ Toolkit are:

   - Caffe
   - TensorFlow
   - MXNet
   - ONNX (which can support PyTorch and Apple ML models through another conversion step)
   - Kaldi
   
These are all open source, just like the OpenVINO™ Toolkit. Caffe is originally from UC Berkeley, TensorFlow is from Google Brain, MXNet is from Apache Software, ONNX is combined effort of Facebook and Microsoft, and Kaldi was originally an individual’s effort. Most of these are fairly multi-purpose frameworks, while Kaldi is primarily focused on speech recognition data.

There are some differences in how exactly to handle these, although most differences are handled under the hood of the OpenVINO™ Toolkit. For example, TensorFlow has some different steps for certain models, or frozen vs. unfrozen models. However, most of the functionality is shared across all of the supported frameworks.

## Intermediate Representations

Intermediate Representations (IRs) are the OpenVINO™ Toolkit’s standard structure and naming for neural network architectures. A `Conv2D` layer in TensorFlow, `Convolution` layer in Caffe, or Conv layer in ONNX are all converted into a `Convolution` layer in an IR.

The IR is able to be loaded directly into the Inference Engine, and is actually made of two output files from the Model Optimizer: an XML file and a binary file. The XML file holds the model architecture and other important metadata, while the binary file holds weights and biases in a binary format. You need both of these files in order to run inference Any desired optimizations will have occurred while this is generated by the Model Optimizer, such as changes to precision. You can generate certain precisions with the `--data_type` argument, which is usually FP32 by default.

## Using the Model Optimizer with TensorFlow Models

Once the Model Optimizer is configured, the next thing to do with a TensorFlow model is to determine whether to use a frozen or unfrozen model. You can either freeze your model, which I would suggest, or use the separate instructions in the documentation to convert a non-frozen model. Some models in TensorFlow may already be frozen for you, so you can skip this step.

From there, you can feed the model into the Model Optimizer, and get your Intermediate Representation. However, there may be a few items specific to TensorFlow for that stage, which you’ll need to feed into the Model Optimizer before it can create an IR for use with the Inference Engine.

TensorFlow models can vary for what additional steps are needed by model type, being unfrozen or frozen, or being from the TensorFlow Detection Model Zoo. Unfrozen models usually need the `--mean_values` and `--scale` parameters fed to the Model Optimizer, while the frozen models from the Object Detection Model Zoo don’t need those parameters. However, the frozen models will need TensorFlow-specific parameters like `--tensorflow_use_custom_operations_config` and `--tensorflow_object_detection_api_pipeline_config`. Also, `--reverse_input_channels` is usually needed, as TF model zoo models are trained on RGB images, while OpenCV usually loads as BGR. Certain models, like YOLO, DeepSpeech, and more, have their own separate pages.

### TensorFlow Object Detection Model Zoo

The models in the TensorFlow Object Detection Model Zoo can be used to even further extend the pre-trained models available to you. These are in TensorFlow format, so they will need to be fed to the Model Optimizer to get an IR. The models are just focused on object detection with bounding boxes, but there are plenty of different model architectures available.

## Solution: Convert a TensorFlow Model

Here's what I entered to convert the SSD MobileNet V2 model from TensorFlow:
`
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
`
This is pretty long! I would suggest considering setting a path environment variable for the Model Optimizer if you are working locally on a Linux-based machine. You could do something like this:
`export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer`

And then when you need to use it, you can utilize it with $MOD_OPT/mo.py instead of entering the full long path each time. In this case, that would also help shorten the path to the ssd_v2_support.json file used.

## Using the Model Optimizer with Caffe Models

The process for converting a Caffe model is fairly similar to the TensorFlow one, although there’s nothing about freezing the model this time around, since that’s a TensorFlow concept. Caffe does have some differences in the set of supported model architectures. Additionally, Caffe models need to feed both the `.caffemodel` file, as well as a `.prototxt` file, into the Model Optimizer. If they have the same name, only the model needs to be directly input as an argument, while if the `.prototxt` file has a different name than the model, it should be fed in with `--input_proto` as well.

## Solution: Convert a Caffe Model

Here's what I entered to convert the Squeezenet V1.1 model from Caffe:
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt`

## Using the Model Optimizer with ONNX Models

The process for converting an ONNX model is again quite similar to the previous two, although ONNX does not have any ONNX-specific arguments to the Model Optimizer. So, you’ll only have the general arguments for items like changing the precision.

Additionally, if you are working with PyTorch or Apple ML models, they need to be converted to ONNX format first, which is done outside of the OpenVINO™ Toolkit. See the link further down on this page if you are interested in doing so.

## Using the Model Optimizer with ONNX Models

Here's what I entered to convert the AlexNet model from ONNX:
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx`

## Cutting Parts of a Model

Cutting a model is mostly applicable for TensorFlow models. As we saw earlier in converting these models, they sometimes have some extra complexities. Some common reasons for cutting are:

   -The model has pre- or post-processing parts that don’t translate to existing Inference Engine layers.
   -The model has a training part that is convenient to be kept in the model, but is not used during inference.
   -The model is too complex with many unsupported operations, so the complete model cannot be converted in one shot.
   -The model is one of the supported SSD models. In this case, you need to cut a post-processing part off.
   -There could be a problem with model conversion in the Model Optimizer or with inference in the Inference Engine. To localize the issue, cutting the model could help to find the problem

There’s two main command line arguments to use for cutting a model with the Model Optimizer, named intuitively as `--input` and `--output`, where they are used to feed in the layer names that should be either the new entry or exit points of the model.

## Supported Layers

Earlier, we saw some of the supported layers when looking at the names when converting from a supported framework to an IR. While that list is useful for one-offs, you probably don’t want to check whether each and every layer in your model is supported. You can also just see when you run the Model Optimizer what will convert.

What happens when a layer isn’t supported by the Model Optimizer? One potential solution is the use of custom layers, which we’ll go into more shortly. Another solution is actually running the given unsupported layer in its original framework. For example, you could potentially use TensorFlow to load and process the inputs and outputs for a specific layer you built in that framework, if it isn’t supported with the Model Optimizer. Lastly, there are also unsupported layers for certain hardware, that you may run into when working with the Inference Engine. In this case, there are sometimes extensions available that can add support. We’ll discuss that approach more in the next lesson.

## Custom Layers

Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit, although you shouldn’t have to use it very often, if at all, due to all of the supported layers. However, it’s useful to know a little about its existence and how to use it if the need arises.

The list of supported layers from earlier very directly relates to whether a given layer is a custom layer. Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.

To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

## Glossary

### Model Optimizer

A command-line tool used for converting a model from one of the supported frameworks to an Intermediate Representation (IR), including certain performance optimizations, that is compatible with the Inference Engine.


### Optimization Techniques

Optimization techniques adjust the original trained model in order to either reduce the size of or increase the speed of a model in performing inference. Techniques discussed in the lesson include quantization, freezing and fusion.


### Quantization

Reduces precision of weights and biases (to lower precision floating point values or integers), thereby reducing compute time and size with some (often minimal) loss of accuracy.


### Freezing

In TensorFlow this removes metadata only needed for training, as well as converting variables to constants. Also a term in training neural networks, where it often refers to freezing layers themselves in order to fine tune only a subset of layers.


### Fusion

The process of combining certain operations together into one operation and thereby needing less computational overhead. For example, a batch normalization layer, activation layer, and convolutional layer could be combined into a single operation. This can be particularly useful for GPU inference, where the separate operations may occur on separate GPU kernels, while a fused operation occurs on one kernel, thereby incurring less overhead in switching from one kernel to the next.


### Supported Frameworks

The Intel® Distribution of OpenVINO™ Toolkit currently supports models from five frameworks (which themselves may support additional model frameworks): Caffe, TensorFlow, MXNet, ONNX, and Kaldi.

### Caffe

The “Convolutional Architecture for Fast Feature Embedding” (CAFFE) framework is an open-source deep learning library originally built at UC Berkeley.

### TensorFlow

TensorFlow is an open-source deep learning library originally built at Google. As an Easter egg for anyone who has read this far into the glossary, this was also your instructor’s first deep learning framework they learned, back in 2016 (pre-V1!).

### MXNet

Apache MXNet is an open-source deep learning library built by Apache Software Foundation.

### ONNX

The “Open Neural Network Exchange” (ONNX) framework is an open-source deep learning library originally built by Facebook and Microsoft. PyTorch and Apple-ML models are able to be converted to ONNX models.

### Kaldi

While still open-source like the other supported frameworks, Kaldi is mostly focused around speech recognition data, with the others being more generalized frameworks.

### Intermediate Representation

A set of files converted from one of the supported frameworks, or available as one of the Pre-Trained Models. This has been optimized for inference through the Inference Engine, and may be at one of several different precision levels. Made of two files:

   - `.xml` - Describes the network topology
   - `.bin` - Contains the weights and biases in a binary file

### Supported Layers

Layers supported for direct conversion from supported framework layers to intermediate representation layers through the Model Optimizer. While nearly every layer you will ever use is in the supported frameworks is supported, there is sometimes a need for handling Custom Layers.

### Custom Layers

Custom layers are those outside of the list of known, supported layers, and are typically a rare exception. Handling custom layers in a neural network for use with the Model Optimizer depends somewhat on the framework used; other than adding the custom layer as an extension, you otherwise have to follow instructions specific to the framework.


-----------------------------------------------------------------------------------------------------------

# Lesson 4 - The Inference Engine

In this lesson we'll cover:

   - Basics of the Inference Engine
   - Supported Devices
   - Feeding an Intermediate Representation to the Inference Engine
   - Making Inference Requests
   - Handling Results from the Inference Engine
   - Integrating the Inference Model into an App
   
## The Inference Engine

The Inference Engine runs the actual inference on a model. It only works with the Intermediate Representations that come from the Model Optimizer, or the Intel® Pre-Trained Models in OpenVINO™ that are already in IR format.

Where the Model Optimizer made some improvements to size and complexity of the models to improve memory and computation times, the Inference Engine provides hardware-based optimizations to get even further improvements from a model. This really empowers your application to run at the edge and use up as little of device resources as possible.

The Inference Engine has a straightforward API to allow easy integration into your edge application. The Inference Engine itself is actually built in C++ (at least for the CPU version), leading to overall faster operations, However, it is very common to utilize the built-in Python wrapper to interact with it in Python code.

## Supported Devices

The supported devices for the Inference Engine are all Intel® hardware, and are a variety of such devices: CPUs, including integrated graphics processors, GPUs, FPGAs, and VPUs. You likely know what CPUs and GPUs are already, but maybe not the others.

FPGAs, or Field Programmable Gate Arrays, are able to be further configured by a customer after manufacturing. Hence the “field programmable” part of the name.

VPUs, or Vision Processing Units, are going to be like the Intel® Neural Compute Stick. They are small, but powerful devices that can be plugged into other hardware, for the specific purpose of accelerating computer vision tasks.

### Differences Among Hardware

Mostly, how the Inference Engine operates on one device will be the same as other supported devices; however, you may remember me mentioning a CPU extension in the last lesson. That’s one difference, that a CPU extension can be added to support additional layers when the Inference Engine is used on a CPU.

There are also some differences among supported layers by device, which is linked to at the bottom of this page. Another important one to note is regarding when you use an Intel® Neural Compute Stick (NCS). An easy, fairly low-cost method of testing out an edge app locally, outside of your own computer is to use the NCS2 with a Raspberry Pi. The Model Optimizer is not supported directly with this combination, so you may need to create an Intermediate Representation on another system first, although there are some instructions for one way to do so on-device. The Inference Engine itself is still supported with this combination.

## Using the Inference Engine with an IR

`IECore` and `IENetwork`

To load an IR into the Inference Engine, you’ll mostly work with two classes in the `openvino.inference_engine` library (if using Python):

   - `IECore`, which is the Python wrapper to work with the Inference Engine
   - `IENetwork`, which is what will initially hold the network and get loaded into IECore

The next step after importing is to set a couple variables to actually use the IECore and IENetwork. In the IECore documentation, no arguments are needed to initialize. To use IENetwork, you need to load arguments named `model` and `weights` to initialize - the XML and Binary files that make up the model’s Intermediate Representation.

### Check Supported Layers

In the IECore documentation, there was another function called `query_network`, which takes in an IENetwork as an argument and a device name, and returns a list of layers the Inference Engine supports. You can then iterate through the layers in the IENetwork you created, and check whether they are in the supported layers list. If a layer was not supported, a CPU extension may be able to help.

The `device_name` argument is just a string for which device is being used - `”CPU”`, `”GPU”`, `”FPGA”`, or `”MYRIAD”` (which applies for the Neural Compute Stick).

### CPU extension

If layers were successfully built into an Intermediate Representation with the Model Optimizer, some may still be unsupported by default with the Inference Engine when run on a CPU. However, there is likely support for them using one of the available CPU extensions.

These do differ by operating system a bit, although they should still be in the same overall location. If you navigate to your OpenVINO™ install directory, then `deployment_tools`, `inference_engine`, `lib`, `intel64`:
 - On Linux, you’ll see a few CPU extension files available for AVX and SSE. That’s a bit outside of the scope of the course, but look up Advanced Vector Extensions if you want to know more there. In the classroom workspace, the SSE file will work fine.
 
You can add these directly to the `IECore` using their full path. After you’ve added the CPU extension, if necessary, you should re-check that all layers are now supported. If they are, it’s finally time to load the model into the IECore.

## Solution: Feed an IR to the Inference Engine

First, add the additional libraries (`os` may not be needed depending on how you get the model file names):

``` python
### Load the necessary libraries
import os
from openvino.inference_engine import IENetwork, IECore
```
Then, to load the Intermediate Representation and feed it to the Inference Engine:

``` python
def load_to_IE(model_xml):
    ### Load the Inference Engine API
    plugin = IECore()

    ### Load IR files into their related class
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)

    ### Add a CPU extension, if applicable.
    plugin.add_extension(CPU_EXTENSION, "CPU")

    ### Get the supported layers of the network
    supported_layers = plugin.query_network(network=net, device_name="CPU")

    ### Check for any unsupported layers, and let the user
    ### know if anything is missing. Exit the program, if so.
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        print("Check whether extensions are available to add to IECore.")
        exit(1)

    ### Load the network into the Inference Engine
    plugin.load_network(net, "CPU")

    print("IR successfully loaded into Inference Engine.")

    return
```
Note that a more optimal approach here would actually check whether a CPU extension was added as an argument by the user, but to keep things simple, I hard-coded it for the exercise.

### Running Your Implementation

You should make sure your implementation runs with all three pre-trained models we worked with earlier (and you are welcome to also try the models you converted in the previous lesson from TensorFlow, Caffe and ONNX, although your workspace may not have these stored). I placed these in the `/home/workspace/models` directory for easier use, and because the workspace will reset the `/opt` directory between sessions.

#### Human Pose Estimation
``` python feed_network.py -m /home/workspace/models/human-pose-estimation-0001.xml ```

You can run the other two by updating the model name in the above.

#### Text Detection:
``` python feed_network.py -m /home/workspace/models/text-detection-0004.xml ```

#### Determining Car Type & Color
``` python feed_network.py -m /home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml ```


## Sending Inference Requests to the IE

After you load the `IENetwork` into the `IECore`, you get back an `ExecutableNetwork`, which is what you will send inference requests to. There are two types of inference requests you can make: Synchronous and Asynchronous. There is an important difference between the two on whether your app sits and waits for the inference or can continue and do other tasks.

With an `ExecutableNetwork`, synchronous requests just use the `infer` function, while asynchronous requests begin with `start_async`, and then you can `wait` until the request is complete. These requests are `InferRequest` objects, which will hold both the input and output of the request.

## Asynchronous Requests

### Synchronous

Synchronous requests will wait and do nothing else until the inference response is returned, blocking the main thread. In this case, only one frame is being processed at once, and the next frame cannot be gathered until the current frame’s inference request is complete.

### Asynchronous

You may have heard of asynchronous if you do front-end or networking work. In that case, you want to process things asynchronously, so in case the response for a particular item takes a long time, you don’t hold up the rest of your website or app from loading or operating appropriately.

Asynchronous, in our case, means other tasks may continue while waiting on the IE to respond. This is helpful when you want other things to still occur, so that the app is not completely frozen by the request if the response hangs for a bit.

Where the main thread was blocked in synchronous, asynchronous does not block the main thread. So, you could have a frame sent for inference, while still gathering and pre-processing the next frame. You can make use of the "wait" process to wait for the inference result to be available.

You could also use this with multiple webcams, so that the app could "grab" a new frame from one webcam while performing inference for the other.

## Solution: Inference Requests

### Synchronous Solution

``` python
def sync_inference(exec_net, input_blob, image):
    '''
    Performs synchronous inference
    Return the result of inference
    '''
    result = exec_net.infer({input_blob: image})

    return result
```

### Asynchronous Solution

``` python
def async_inference(exec_net, input_blob, image):
    '''
    Performs asynchronous inference
    Returns the `exec_net`
    '''
    exec_net.start_async(request_id=0, inputs={input_blob: image})
    while True:
        status = exec_net.requests[0].wait(-1)
        if status == 0:
            break
        else:
            time.sleep(1)
    return exec_net
```

I don't actually need `time.sleep()` here - using the `-1` with `wait()` is able to perform similar functionality.

### Testing

You can run the test file to check your implementations using inference on multiple models.
```python test.py```

## Handling Results

You saw at the end of the previous exercise that the inference requests are stored in a `requests` attribute in the `ExecutableNetwork`. There, we focused on the fact that the `InferRequest` object had a `wait` function for asynchronous requests.

Each `InferRequest` also has a few attributes - namely, `inputs`, `outputs` and `latency`. As the names suggest, `inputs` in our case would be an image frame, `outputs` contains the results, and `latency` notes the inference time of the current request, although we won’t worry about that right now.

It may be useful for you to print out exactly what the `outputs` attribute contains after a request is complete. For now, you can ask it for the `data` under the `“prob”` key, or sometimes `output_blob`, to get an array of the probabilities returned from the inference request.

Note: 
- An ExecutableNetwork contains an InferRequest attribute by the name of requests, and feeding a given request ID key to this attribute will get the specific inference request in question.
- From within this InferRequest object, it has an attribute of outputs from which you can use your output_blob to get the results of that inference request.


## Solution: Integrate into an App

Note: There is one small change from the code on-screen for running on Linux machines versus Mac. On Mac, `cv2.VideoWriter` uses `cv2.VideoWriter_fourcc('M','J','P','G')` to write an `.mp4` file, while Linux uses `0x00000021`.

### Functions in `inference.py`

I covered the `async` and `wait` functions here as it's split out slightly differently than we saw in the last exercise.

First, it's important to note that output and input blobs were grabbed higher above when the network model is loaded:

``` python
self.input_blob = next(iter(self.network.inputs))
self.output_blob = next(iter(self.network.outputs))
```
From there, you can mostly use similar code to before:

``` python
   def async_inference(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=0, 
            inputs={self.input_blob: image})
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[0].wait(-1)
        return status
```

You can grab the network output using the appropriate `request` with the `output_blob` key:

``` python
    def extract_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs[self.output_blob]
```
### `app.py`

The next steps in `app.py`, before customization, are largely based on using the functions in `inference.py`:

``` python
 ### Initialize the Inference Engine
    plugin = Network()

    ### Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    ...

        ### Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### Perform inference on the frame
        plugin.async_inference(p_frame)

        ### Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            out.write(frame)
```

The `draw_boxes` function is used to extract the bounding boxes and draw them back onto the input image.

``` python
def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame
```

### Customizing `app.py`

Adding the customization only took a few extra steps. 
#### Parsing the command line arguments
First, you need to add the additional command line arguments:

``` python
 c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
 ct_desc = "The confidence threshold to use with the bounding boxes"

    ...

    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
```

The names and descriptions here, and even how you use the default values, can be up to you.

#### Handling new arguments

I needed to also process these arguments a little further. This is pretty open based on your own implementation - since I took in a color string, I need to convert it to a BGR tuple for use as a OpenCV colors.

``` python
def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']
```

I then need to call this with the related argument, as well as make sure the confidence threshold argument is a float value.

``` python
args.c = convert_color(args.c)
args.ct = float(args.ct)
```

#### Adding customization to `draw_boxes()`
The final step was to integrate these new arguments into my draw_boxes() function. I needed to make sure that the arguments are fed to the function:

``` python
frame = draw_boxes(frame, result, args, width, height) 
```
and then I can use them where appropriate in the updated function.

``` python
def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
    return frame
```

With everything implemented, I could run my app as such (given I re-used the previously converted TF model from the Model Optimizer lesson) if I wanted blue bounding boxes and a confidence threshold of `0.6`: <br/>
```python app.py -m frozen_inference_graph.xml -ct 0.6 -c BLUE```

## Lesson Glossary

### Inference Engine

Provides a library of computer vision functions, supports calls to other computer vision libraries such as OpenCV, and performs optimized inference on Intermediate Representation models. Works with various plugins specific to different hardware to support even further optimizations.

### Synchronous

Such requests wait for a given request to be fulfilled prior to continuing on to the next request.

### Asynchronous

Such requests can happen simultaneously, so that the start of the next request does not need to wait on the completion of the previous.

### IECore

The main Python wrapper for working with the Inference Engine. Also used to load an IENetwork, check the supported layers of a given network, as well as add any necessary CPU extensions.

### IENetwork

A class to hold a model loaded from an Intermediate Representation (IR). This can then be loaded into an IECore and returned as an Executable Network.

### ExecutableNetwork

An instance of a network loaded into an IECore and ready for inference. It is capable of both synchronous and asynchronous requests, and holds a tuple of InferRequest objects.

### InferRequest

Individual inference requests, such as image by image, to the Inference Engine. Each of these contain their inputs as well as the outputs of the inference request once complete.

-----------------------------------------------------------------------------------------------------------

# Lesson 5 - Deploying an Edge App

In this lesson we'll cover:

   - Basics of OpenCV
   - Handling Input Streams in OpenCV
   - Processing Model Outputs for Additional Useful Information
   - The Basics of MQTT and their use with IoT devices
   - Sending statistics and video streams to a server
   - Performance basics
   - And finish up by thinking about additional model use cases, as well as end user needs
    
## OpenCV Basics

OpenCV is an open-source library for various image processing and computer vision techniques that runs on a highly optimized C++ back-end, although it is available for use with Python and Java as well. It’s often helpful as part of your overall edge applications, whether using it’s built-in computer vision techniques or handling image processing.

### Uses of OpenCV

There’s a lot of uses of OpenCV. In your case, you’ll largely focus on its ability to capture and read frames from video streams, as well as different pre-processing techniques, such as resizing an image to the expected input size of your model. It also has other pre-processing techniques like converting from one color space to another, which may help in extracting certain features from a frame. There are also plenty of computer vision techniques included, such as Canny Edge detection, which helps to extract edges from an image, and it extends even to a suite of different machine learning classifiers for tasks like face detection.

### Useful OpenCV functions

- `VideoCapture` - can read in a video or image and extract a frame from it for processing
- `resize` is used to resize a given frame
- `cvtColor` can convert between color spaces.
   - You may remember from awhile back that TensorFlow models are usually trained with RGB images, while OpenCV is going to load frames as BGR. There was a technique with the Model Optimizer that would build the TensorFlow model to appropriately handle BGR. If you did not add that additional argument at the time, you could use this function to convert each image to RGB, but that’s going to add a little extra processing time.
- `rectangle` - useful for drawing bounding boxes onto an output image
- `imwrite` - useful for saving down a given image

## Handling Input Streams

Being able to efficiently handle video files, image files, or webcam streams is an important part of an edge application. If I were to be running the webcam on my Macbook for instance and performing inference, a surprisingly large amount of resources get used up simply to use the webcam. That’s why it’s useful to utilize the OpenCV functions built for this - they are about as optimized for general use with input streams as you will find.

### Open & Read A Video

We saw the `cv2.VideoCapture` function in the previous video. This function takes either a zero for webcam use, or the path to the input image or video file. That’s just the first step, though. This “capture” object must then be opened with `capture.open`.

Then, you can basically make a loop by checking if `capture.isOpened`, and you can read a frame from it with `capture.read`. This `read` function can actually return two items, a boolean and the frame. If the boolean is false, there’s no further frames to read, such as if the video is over, so you should break out of the loop.

### Closing the Capture

Once there are no more frames left to capture, there’s a couple of extra steps to end the process with OpenCV.

- First, you’ll need to `release` the capture, which will allow OpenCV to release the captured file or stream
- Second, you’ll likely want to use `cv2.destroyAllWindows`. This will make sure any additional windows, such as those used to view output frames, are closed out
- Additionally, you may want to add a call to `cv2.waitKey` within the loop, and break the loop if your desired key is pressed. For example, if the key pressed is 27, that’s the Escape key on your keyboard - that way, you can close the stream midway through with a single button. Otherwise, you may get stuck with an open window that’s a bit difficult to close on its own.
    
## Solution: Handling Input Streams

<b>Note:</b> There are two small changes from the code on-screen for running on Linux machines versus Mac.

- On Mac, `cv2.VideoWriter` uses `cv2.VideoWriter_fourcc('M','J','P','G')` to write an `.mp4` file, while Linux uses `0x00000021`.
- On Mac, the output with the given code on using Canny Edge Detection will run fine. However, on Linux, you'll need to use `np.dstack` to make a 3-channel array to write back to the out file, or else the video won't be able to be opened correctly: 
``` python
frame = np.dstack((frame, frame, frame))
```

Let's walk through each of the tasks.

   > Implement a function that can handle camera image, video file or webcam inputs

The main thing here is just to check the `input` argument passed to the command line.

This will differ by application, but in this implementation, the argument parser makes note that `"CAM"` is an acceptable input meaning to use the webcam. In that case, the `input_stream` or `args.i` should be set to `0`, as `cv2.VideoCapture()` can use the system camera when set to zero.

The next is checking whether the input name is a filepath containing an image file type, such as `.jpg` or `.png`. If so, you'll just set the `input_stream` to that path. You should also set the flag here to note it is a single image, so you can save down the image as part of one of the later steps.

The last one is for a video file. It's mostly the same as the image, as the `input_stream` is the filepath passed to the `input` argument, but you don't need to use a flag here.

A last thing you should consider in your app here is exception handling - does your app just crash if the input is invalid or missing, or does it still log useful information to the user?

   > Use cv2.VideoCapture() and open the capture stream

``` python
capture = cv2.VideoCapture(input_stream)
capture.open(args.input)

while capture.isOpened():
    flag, frame = cap.read()
    if not flag:
        break
```

It's a bit outside of the instructions, but it's also important to check whether a key gets pressed within the while loop, to make it easier to exit.

You can use:

``` python
key_pressed = cv2.waitKey(60)
```

to check for a key press, and then

``` python
if key_pressed == 27:
    break
```

to break the loop, if needed. Key 27 is the Escape button.

  >  Re-size the frame to 100x100

``` python 
image = cv2.resize(frame, (100, 100))
```

  >  Add Canny Edge Detection to the frame with min & max values of 100 and 200, respectively

Canny Edge detection is useful for detecting edges in an image, and has been a useful computer vision technique for extracting features. This was a step just so you could get a little more practice with OpenCV.

``` python
edges = cv2.Canny(image,100,200)
```

  >  Display the resulting frame if it's video, or save it if it is an image

For video:

``` python
cv2.imshow('display', edges)
```

For a single image:

``` python
cv2.imwrite('output.jpg', edges)
```

  >  Close the stream and any windows at the end of the application

Make sure to close your windows here so you don't get stuck with them on-screen.

```python
capture.release()
cv2.destroyAllWindows()
```

### Testing the Implementation

I can then test both an image and a video with the following:

```python app.py -i blue-car.jpg```

```python app.py -i test_video.mp4```


## Gathering Useful Information from Model Outputs

Training neural networks focuses a lot on accuracy, such as detecting the right bounding boxes and having them placed in the right spot. But what should you actually do with bounding boxes, semantic masks, classes, etc.? How would a self-driving car make a decision about where to drive based solely off the semantic classes in an image?

It’s important to get useful information from your model - information from one model could even be further used in an additional model, such as traffic data from one set of days being used to predict traffic on another set of days, such as near to a sporting event.

For the traffic example, you’d likely want to count how many bounding boxes there are, but also make sure to only count once for each vehicle until it leaves the screen. You could also consider which part of the screen they come from, and which part they exit from. Does the left turn arrow need to last longer near to a big event, as all the cars seem to be heading in that direction?

In an earlier exercise, you played around a bit with the confidence threshold of bounding box detections. That’s another way to extract useful statistics - are you making sure to throw out low confidence predictions?

## Solution: Process Model Outputs

My approach in this exercise was to check if the bad combination of pets was on screen, but also to track whether I already warned them in the current incident. Now, I might also consider re-playing the warning after a certain time period in a single consecutive incident, but the provided video file does not really have that long of consecutive timespans. I also output a "timestamp" by checking how many frames had been processed so far at 30 fps.

Before the video loop, I added:

``` python
counter = 0
incident_flag = False
```

Within the loop, after a frame is read, I make sure to increment the counter: `counter+=1`.

I made an `assess_scene` function for most of the processing:

``` python
def assess_scene(result, counter, incident_flag):
    '''
    Based on the determined situation, potentially send
    a message to the pets to break it up.
    '''
    if result[0][1] == 1 and not incident_flag:
        timestamp = counter / 30
        print("Log: Incident at {:.2f} seconds.".format(timestamp))
        print("Break it up!")
        incident_flag = True
    elif result[0][1] != 1:
        incident_flag = False
    return incident_flag
```

And I call that within the loop right after the result is available:

``` python
incident_flag = assess_scene(result, counter, incident_flag)
```

### Running the App

To run the app, I just used:

```python app.py -m model.xml```

Since the model was provided here in the same directory.


## Intro to MQTT

### MQTT

MQTT stands for MQ Telemetry Transport, where the MQ came from an old IBM product line called IBM MQ for Message Queues (although MQTT itself does not use queues). That doesn’t really give many hints about its use.

MQTT is a lightweight publish/subscribe architecture that is designed for resource-constrained devices and low-bandwidth setups. It is used a lot for Internet of Things devices, or other machine-to-machine communication, and has been around since 1999. Port 1883 is reserved for use with MQTT.

### Publish/Subscribe

In the publish/subscribe architecture, there is a broker, or hub, that receives messages published to it by different clients. The broker then routes the messages to any clients subscribing to those particular messages.

This is managed through the use of what are called “topics”. One client publishes to a topic, while another client subscribes to the topic. The broker handles passing the message from the publishing client on that topic to any subscribers. These clients therefore don’t need to know anything about each other, just the topic they want to publish or subscribe to.

MQTT is one example of this type of architecture, and is very lightweight. While you could publish information such as the count of bounding boxes over MQTT, you cannot publish a video frame using it. Publish/subscribe is also used with self-driving cars, such as with the Robot Operating System, or ROS for short. There, a stop light classifier may publish on one topic, with an intermediate system that determines when to brake subscribing to that topic, and then that system could publish to another topic that the actual brake system itself subscribes to.


## Communicating with MQTT

There is a useful Python library for working with MQTT called `paho-mqtt`. Within, there is a sub-library called `client`, which is how you create an MQTT client that can publish or subscribe to the broker.

To do so, you’ll need to know the IP address of the broker, as well as the port for it. With those, you can connect the client, and then begin to either publish or subscribe to topics.

Publishing involves feeding in the topic name, as well as a dictionary containing a message that is dumped to JSON. Subscribing just involves feeding in the topic name to be subscribed to.

## Streaming Images to a Server

Sometimes, you may still want a video feed to be streamed to a server. A security camera that detects a person where they shouldn’t be and sends an alert is useful, but you likely want to then view the footage. Since MQTT can’t handle images, we have to look elsewhere.

At the start of the course, we noted that network communications can be expensive in cost, bandwidth and power consumption. Video streaming consumes a ton of network resources, as it requires a lot of data to be sent over the network, clogging everything up. Even with high-speed internet, multiple users streaming video can cause things to slow down. As such, it’s important to first consider whether you even need to stream video to a server, or at least only stream it in certain situations, such as when your edge AI algorithm has detected a particular event.

### FFmpeg

Of course, there are certainly situations where streaming video is necessary. The FFmpeg library is one way to do this. The name comes from “fast forward” MPEG, meaning it’s supposed to be a fast way of handling the MPEG video standard (among others).

In our case, we’ll use the `ffserver` feature of FFmpeg, which, similar to MQTT, will actually have an intermediate FFmpeg server that video frames are sent to. The final Node server that displays a webpage will actually get the video from that FFmpeg server.

There are other ways to handle streaming video as well. In Python, you can also use a flask server to do some similar things, although we’ll focus on FFmpeg here.

### Setting up FFmpeg

You can download FFmpeg from ffmpeg.org. Using `ffserver` in particular requires a configuration file that we will provide for you. This config file sets the port and IP address of the server, as well as settings like the ports to receive video from, and the framerate of the video. These settings can also allow it to listen to the system stdout buffer, which is how you can send video frames to it in Python.

### Sending frames to FFmpeg

With the `sys` Python library, can use `sys.stdout.buffer.write(frame)` and `sys.stdout.flush()` to send the `frame` to the ffserver when it is running.

If you have a `ffmpeg` folder containing the configuration file for the server, you can launch the `ffserver` with the following from the command line:

``` sudo ffserver -f ./ffmpeg/server.conf ```

From there, you need to actually pipe the information from the Python script to FFmpeg. To do so, you add the `|` symbol after the python script (as well as being after any related arguments to that script, such as the model file or CPU extension), followed by `ffmpeg` and any of its related arguments.

For example:

```python app.py -m “model.xml” | ffmpeg -framerate 24```

And so on with additional arguments before or after the pipe symbol depending on whether they are for the Python application or for FFmpeg.

### Handling Statistics and Images from a Node Server

Node.js is an open-source environment for servers, where Javascript can be run outside of a browser. Consider a social media page, for instance - that page is going to contain different content for each different user, based on their social network. Node allows for Javascript to run outside of the browser to gather the various relevant posts for each given user, and then send those posts to the browser.

In our case, a Node server can be used to handle the data coming in from the MQTT and FFmpeg servers, and then actually render that content for a web page user interface.

## Solution: Server Communications

<b>Note:</b> You will need to use port 3001 in the workspace for MQTT within app.py instead of the standard port 1883. 
Let's focus on MQTT first, and then FFmpeg.

### MQTT

First, I import the MQTT Python library. I use an alias here so the library is easier to work with.

``` python
import paho.mqtt.client as mqtt
```

I also need to `import socket` so I can connect to the MQTT server. Then, I can get the IP address and set the port for communicating with the MQTT server.

``` python
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
```

This will set the IP address and port, as well as the keep alive interval. The keep alive interval is used so that the server and client will communicate every 60 seconds to confirm their connection is still open, if no other communication (such as the inference statistics) is received.

<b> Note: </b> The port here is 3001, instead of the normal MQTT port of 1883, as our classroom workspace environment only allows ports from 3000-3009 to be used. The real importance is here to make sure this matches to what is set for the MQTT broker server to be listening on, which in this case has also been set to 3001 (you can see this in `config.js` within the MQTT server's files in the workspace).

Connecting to the client can be accomplished with:

``` python
client = mqtt.Client()
client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
```

Note that mqtt in the above was my import alias - if you used something different, that line will also differ slightly, although will still use `Client()`.

The final piece for MQTT is to actually publish the statistics to the connected client.

``` python
topic = "some_string"
client.publish(topic, json.dumps({"stat_name": statistic}))
```

The topic here should match to the relevant topic that is being subscribed to from the other end, while the JSON being published should include the relevant name of the statistic for the node server to parse (with the name like the key of a dictionary), with the statistic passed in with it (like the items of a dictionary).

``` python
client.publish("class", json.dumps({"class_names": class_names}))
client.publish("speedometer", json.dumps({"speed": speed}))
```

And, at the end of processing the input stream, make sure to disconnect.

``` python
client.disconnect()
```

### FFmpeg

FFmpeg does not actually have any real specific imports, although we do want the standard `sys` library

```python 
import sys
```

This is used as the `ffserver` can be configured to read from `sys.stdout`. Once the output frame has been processed (drawing bounding boxes, semantic masks, etc.), you can write the frame to the `stdout` buffer and `flush` it.

``` python
sys.stdout.buffer.write(frame)  
sys.stdout.flush()
```

And that's it! As long as the MQTT and FFmpeg servers are running and configured appropriately, the information should be able to be received by the final node server, and viewed in the browser.

### Running the App

To run the app itself, with the UI server, MQTT server, and FFmpeg server also running, do:

```python app.py | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm```

This will feed the output of the app to FFmpeg.

## Analyzing Performance Basics

We’ve talked a lot about optimizing for inference and running apps at the edge, but it’s important not to skip past the accuracy of your edge AI model. Lighter, quicker models are helpful for the edge, and certain optimizations like lower precision that help with these will have some impact on accuracy, as we discussed earlier on.

No amount of skillful post-processing and attempting to extract useful data from the output will make up for a poor model choice, or one where too many sacrifices were made for speed.

Of course, it all depends on the exact application as to how much loss of accuracy is acceptable. Detecting a pet getting into the trash likely can handle less accuracy than a self-driving car in determining where objects are on the road.

The considerations of speed, size and network impacts are still very important for AI at the Edge. Faster models can free up computation for other tasks, lead to less power usage, or allow for use of cheaper hardware. Smaller models can also free up memory for other tasks, or allow for devices with less memory to begin with. We’ve also discussed some of the network impacts earlier. Especially for remote edge devices, the power costs of heavy network communication may significantly hamper their use,

Lastly, there can be other differences in cloud vs edge costs other than just network effects. While potentially lower up front, cloud storage and computation costs can add up over time. Data sent to the cloud could be intercepted along the way. Whether this is better or not at the edge does depend on a secure edge device, which isn’t always the case for IoT.

## Model Use Cases

It’s important to think about any additional use cases for a given model or application you build, which can reach far beyond the original training set or intended use. For example, object detection can be used for so many things, and focusing on certain classes along with some post-processing can lead to very different applications.

## Concerning End User Needs

If you are building an app for certain end users, it’s very important to consider their needs. Knowing their needs can inform the various trade-offs you make regarding model decisions (speed, size, accuracy, etc.), what information to send to servers, security of information, etc. If they have more resources available, you might go for a higher accuracy but more resource-intensive app, while an end user with remote, low-power devices will likely have to sacrifice some accuracy for a lighter, faster app, and need some additional considerations about network usage.

This is just to get you thinking - building edge applications is about more than just models and code.

## Glossary

### OpenCV

A computer vision (CV) library filled with many different computer vision functions and other useful image and video processing and handling capabilities.

### MQTT

A publisher-subscriber protocol often used for IoT devices due to its lightweight nature. The paho-mqtt library is a common way of working with MQTT in Python.

### Publish-Subscribe Architecture

A messaging architecture whereby it is made up of publishers, that send messages to some central broker, without knowing of the subscribers themselves. These messages can be posted on some given “topic”, which the subscribers can then listen to without having to know the publisher itself, just the “topic”.

### Publisher

In a publish-subscribe architecture, the entity that is sending data to a broker on a certain “topic”.

### Subscriber

In a publish-subscribe architecture, the entity that is listening to data on a certain “topic” from a broker.

### Topic

In a publish-subscribe architecture, data is published to a given topic, and subscribers to that topic can then receive that data.

### FFmpeg

Software that can help convert or stream audio and video. In the course, the related ffserver software is used to stream to a web server, which can then be queried by a Node server for viewing in a web browser.

### Flask

A Python framework useful for web development and another potential option for video streaming to a web browser.

### Node Server

A web server built with Node.js that can handle HTTP requests and/or serve up a webpage for viewing in a browser.

-----------------------------------------------------------------------------------------------------------

You’ve accomplished something amazing! You went from the basics of AI at the Edge, built your skills with pre-trained models, the Model Optimizer, Inference Engine and more with the Intel® Distribution of OpenVINO™ Toolkit, and even learned more about deploying an app at the edge. Best of luck on the project, and I look forward to seeing what you build next!
Intel® DevMesh

Check out the [Intel® DevMesh](https://devmesh.intel.com) website for some more awesome projects others have built, join in on existing projects, or even post some of your own!
Continuing with the Toolkit

If you want to learn more about OpenVINO, you can download the toolkit here: [Download](https://software.intel.com/en-us/openvino-toolkit/choose-download)










