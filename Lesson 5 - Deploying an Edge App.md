# L5 - Deploying an Edge App

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

