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
