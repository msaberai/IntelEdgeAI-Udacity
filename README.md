# Intel® Edge AI Scholarship Program
Build and deploy AI models at the edge! Leverage the potential of edge computing using the Intel® Distribution of OpenVINO™ toolkit to fast-track development of high-performance computer vision and deep learning inference applications. Apply to the Intel® Edge AI Fundamentals Course for the chance to participate in a vibrant student community and to earn one of 750 Nanodegree program scholarships to continue your education with the Intel Distribution of OpenVINO toolkit and other computer vision tools.

## How it works
This scholarship is open to all applicants interested in learning how to work with computer vision deep learning models at the edge. Applicants 18 years of age or older are invited to apply.

We’ll review all applications and select recipients to participate in the Intel® Edge AI Fundamentals Course. This is where your learning begins! Recipients will spend 2.5 months optimizing powerful computer vision deep learning models. The initial fundamentals course is intended for students with some background in AI and computer vision, with experience in either Python or C++ and familiarity with command line basics.

Top students from the initial fundamentals course will be selected for one of 750 follow-up scholarships to the brand-new Intel® Edge AI for IoT Developers Nanodegree program.

### --------------------------------------------------------------------------------------------------
# L1 - Introduction to AI at the Edge
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

### ---------------------------------------------------------------------------------------------------
# L2 - Leveraging Pre-Trained Models

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

### -------------------------------------------------------------------------------------
# L3 - The Model Optimizer

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











