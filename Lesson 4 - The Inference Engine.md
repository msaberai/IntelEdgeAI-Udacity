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

