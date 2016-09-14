---
layout: post
title:  Investigating Loss and Learning
date:   2016-03-13 09:00:54 -0800
categories: machine learning
---

##### I recommend checking out my more recent content, such as this post [here](/machine/learning/2016/09/10/winning-a-loss-ing-battle.html)

## Recap

In my [last post](/machine/learning/2016/02/29/tf-start.html) I talked about how I wanted to learn Tensorflow and machine learning by trying to make a grayscale picture colorizer using deep convolutional neural networks. I found out that that was a challenging problem given my lack of machine learning experience, so I decided to tackle the simpler problem of making an autoencoder. Autoencoders are models that recreate their input and can be used to learn features of the input.

I also had trouble with training my network. The accuracy would jump around but it wouldn’t go down. I found out that inverting my loss function (euclidean distance between each pixel color) made my network train. That went against intuition because I would have expected that training with the inverse would mean it would try to maximize the distance between the pictures.

## New Data

I decided to up the challenge for the network so I generated a new dataset. This time I chose to draw two circles in each image. One white, one gray, and black background. Also I made the images a bit larger at 100x150 pixels. Now the network has to learn more than just the two colors - white and black - that it did before. Here are a bunch of sample images from the dataset:

![Challenging polkadots](/assets/tf_2/dataset_sample.png)

## Looking into the network

I figured a good way to learn what’s going on is to look into the network as it’s training. So I decided to train it on my new dataset and take small snapshots every so often while training. Plus it’s easy to visualize what’s going on inside the network because each layer is 100x150 pixels by however many features that layer had. So for a convolution layer with 16 features we could get 16 100x150 pictures.

At first I trained my network and grabbed one of the features from the first and fourth layer as well as the output of the network. I then turned all of the images I got during training, saved them, then converted them into an animated gif. So the three images below on the right are animated. The rate that the network trains slows down near the end so the images might look static. Give them a few seconds to reset.

Input | Layer 1 Feature 1 | Layer 4 Feature 1 | Output
---------- | ---------- | ---------- | ----------
![Input](/assets/tf_2/original_input.png) | ![Layer1Feature1](/assets/tf_2/layer1_single.gif) | ![Layer4Feature1](/assets/tf_2/layer4_single.gif) | ![Output](/assets/tf_2/output.gif)
|||

It looks like I may have chosen poorly with the number of features and layers I picked for the network. The gif of the first layer doesn’t even look like it’s changing. I guess the network decided it didn’t need to use it (a fair decision considering how over-kill the network was). The fourth layer looks pretty interesting. It seems as though it’s learning the edge of the circle. It kinda looks like two crescent moons. 

The output looks pretty interesting. You can see it starts out very streaky, but then quickly takes the shape of the full circle. It looks like the circles are blurred/out of focus.

I’m kind of amazed the network didn’t learn right away to just pass the pixel values from the input image directly to the output. Instead it more or less learned how to blur the input image. I wonder if further training or messing with the model/training algorithm could yield less blurry results. I’m using dropout while training and I wouldn’t be surprised if that helped the network generalize/blur. But first I want to see what’s going on in the rest of the network.

## Going deeper

Well it looked like only taking one feature from the layers didn’t give us useful insight into how the network was learning. So I decided to save the entire layer. So now we have 32 images from the first layer and 16 images from the second layer.

| Layer 1 All Features
|![Layer1Grid](/assets/tf_2/layer1_grid.gif)
|

Layer one honestly doesn’t look too promising. The changes we see during training are pretty minor across the board. A few circles fill in which I assume, but many remain spotty and untouched. Maybe I should focus on training smaller networks rather than jumping in and training six layer behemoths (well it’s a behemoth for my small GPU!).

| Layer 4 All Features
|![Layer4Grid](/assets/tf_2/layer4_grid.gif)
|

Layer four is much more exciting. We can see most layers go through some development. Most settle and become the two circles, but some stay as the crescent moon shape. I’m beginning to believe that the crescent moon shapes aren’t there for edge detection because it seems like there isn’t one that covers the right side of the circle. So my hypothesis is that they’re just side-effects.

## What I learned

I think the biggest lesson I learned here is that my network is really too big. It seems like there are many redundant and unnecessary things going on in it. I definitely suspected this would be the case when I chose to make the network much bigger than necessary for the problem I chose. I think my next few experiments will be with smaller networks. Maybe I’ll try to make an RBM (restricted Boltzmann machine) and possibly build that up to be a deep belief network. I also haven’t really explored pooling (max/mean pooling) yet either. I originally thought doing that wouldn’t be ideal because I’m not doing a classification problem, and pooling might lose some spacial data. But I’m willing to give it a shot, plus it might reduce the number width of the network in beneficial ways.

It might also be interesting to look at the weights of the convolutions and to see how they change during training. By looking into those I could likely see if they were actually detecting edges and learning features or if they were just passing through blurred pixel values. So many different things to try out…