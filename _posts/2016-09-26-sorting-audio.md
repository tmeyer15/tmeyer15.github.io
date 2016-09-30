---
layout: post
title:  Sorting Audio
date:   2016-09-26 09:00:54 -0800
categories: machine learning
---

## Overview

In this post I’ll briefly discuss some of my previous work/posts in which I explored unsupervised machine learning/computer vision. I’ll talk about why I decided to start working with audio for my next project. I’ll go into details of the image puzzle-solving algorithm and how that inspired my audio puzzle-solving algorithm. I’ll then discuss that algorithm, how I implemented it, some problems I ran into along the way, and how I validated it. And I’ll end with some ideas of where I want to take this next.

## Moving to Audio

Up until now most of the Tensorflow work I’ve done has been with images. The two primary datasets I’ve been toying around with for the last few months were MNIST, a collection of tens of thousands of handwritten grayscale digits, and my homespun polkadot image set, which I thankfully retired rather early. After my last post I figured I could continue with my computer-vision theme, and maybe try out something with color images, possibly much larger than the tiny MNIST images.

Instead I decided to devote some time to audio. I’ve had an interest in audio processing for a while. I remember back in my high school electronics class I spent days getting a wav file onto an EEPROM so we could use it in our TTL/LED display recreation of Guitar Hero. And I remember trying and failing to find natural ‘sections’ of songs using k-means clustering and Fourier transforms of the audio sometime back in college. Though I honestly haven’t started reading much into the  state of the art for audio processing/speech recognition. However, soon after I started working with audio, the [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) paper came out and I found that very inspirational. I thought their tree-like dilated convolution structure was really neat. That gave me a boost in confidence that some really cool audio stuff can be done with deep neural networks.

## Deciding what to work on

As I’ve mentioned before, I’m very interested in unsupervised learning. Especially because it is much easier to acquire unlabeled data for unsupervised learning than it is to get labeled data for supervised learning. Unlabeled pictures/images and songs/audio clips are easily found on the internet. I could even use my personal photo library or music library as a data source! The problem is finding the right unsupervised algorithm to use to process all of this data.

In my previous posts I worked with autoencoders. Autoencoders are models that attempt to recreate their input after squeezing that input into a more compressed representation. Because I was working with images I used the euclidean distance between every pixel of the original image and the recreated image as my loss function. However I noticed that using that technique often resulted in blurry output images. Because of this I don’t think the technique would translate very well to audio because I feel like much of the information of audio is high frequency, and if the autoencoder creates ‘blurry’ audio, then it might lose much of its important information. So while autoencoding works decently well for images, I doubt it would translate well to audio. For example, you can encode the information of a smiling face into two dots and a curved line : ) but as far as I know there isn’t a similarly simple encoding of the audio of someone saying the word ‘smile.’ This is all just me guessing here, I haven’t dived deep enough into audio processing/analysis literature to know if there’s much truth behind this thought.

Without autoencoders I had to come up with another form of unsupervised learning. I recently watched a talk that discussed a researcher’s new approach to unsupervised learning with images by creating a network that solves puzzles. Their approach used two networks, an encoder network and a puzzle solving network. The encoder network is similar to the encoder half of an autoencoder. Image goes in, compact representation of that image comes out. The puzzle solving network takes in two image encodings and tells you where it thinks the source images are physically related to each other.

They trained this model by collecting a bunch of large training images. Then they create a bunch of training samples by randomly cutting a square out of each image and splitting that square into nine smaller squares. Then they take the middle square from those nine squares and randomly pick one of the surrounding eight squares. They run both of those square images through the encoder network, one at a time, so they get two image encodings. Finally they run those two image encodings through the puzzle solving network. The puzzle network takes a guess about where (top, right, top-right, bottom-left, etc) the second image should be with respect to the first image. The combination of the puzzle solving network and the encoder network can then be trained as a classification problem, where each class is the guess as to which of the eight directions the second image is relative to the first. Here’s a picture taken from the [paper](https://arxiv.org/abs/1603.09246) that helps explain their sampling/shuffling technique:

![Sorting jigsaw pieces](/assets/tf_4/NorooziEtAl.png)


You might wonder how this technique would be useful, after all they’re just sorting images that were sorted to begin with. Well, in order for the network to be able to sort each image properly, it needs to encode enough information  about each image before it gets sorted. So the encoder network becomes smart at picking out certain details. Say for example we are looking at an image of a face, and the middle square contains just the mouth of the face and the top image contains just the nose. If the encoder network were smart it would realize that when you’re looking at a face, the mouth is directly below the nose. If the network is able to then recognize a mouth and a nose, then it should be able to solve the puzzle, predicting that the nose appears above the mouth and not to the right, bottom-left, top-right, etc.

The usefulness of this network then comes from the encoder network. Once you’re done training, you can chop off the encoder network and use it as a feature detector in another algorithm. In the talk the authors had a great deal of success applying their [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)-like encoder network to classification tasks, such as ImageNet. I was really impressed with their results and how they were cleverly able to turn unsupervised learning into a classification problem.

After learning about this technique I spent a lot of time thinking about what other sorts of problems/data could be learned by solving ‘puzzles.’ My first thought was that you could potentially teach a network how to sort frames from a video. I feel like that might yield some cool results. I’m curious about what information the encoder network would need to encode about each frame in order to sort the frames correctly. Thinking about that also got me thinking that it might be possible to perform unsupervised learning on audio by having a network that correctly orders a set of shuffled audio clips. I had been wanting to mess around with some audio, so I decided to pick this up as my next project.

## Setting up and experimenting

So here’s a basic rundown of how my audio sorting algorithm works:

I first collected a bunch of audio clips. I decided to download a hundred or so EDM (electronic dance music) songs for my dataset. I chose EDM because I figured they often use digitally produced/synthetic sounds/instruments and have a regular beat throughout the music. I thought it might be easier for the algorithm to sort pieces of it. Then I converted all of the mp3s to 16khz, 16 bit wav files. I read in those songs one at a time and generated hundreds of samples from each song. To generate a sample I chose a random point in the song, grabbed a chunk of audio, skipped ahead by a random padding amount, grabbed another same-sized chunk of audio and repeated to get the number of samples I wanted to sort. Then I shuffled those samples and saved both the shuffled audio samples and the shuffled ordering (the labels) as numpy arrays. I then split the arrays into separate training and validation datasets.

I created a neural network that takes in a sample of audio and turns it into a compact representation. Then I created a neural network that takes in multiple encoded samples (so multiple outputs of the first network) and outputs multiple softmax guesses about the order of the input samples.

At this point you might be wondering if this un-shuffling task is even feasible. Well you can take a listen to the shuffled samples that the network gets here:

[Shuffled audio](/assets/tf_4/shuffled.mp3)

And the neural network needs to find a way to turn that into this:

[Unshuffled audio](/assets/tf_4/unshuffled.mp3)

These were created using 20 0.2 second samples with 0.005 seconds of padding between samples of the song Something. Notice how the padding is almost imperceptible. As you could imagine, the un-shuffling network would really have to understand the higher level structure of the audio in order to sort that clip.

As you could imagine, there are tons of variables and parameters to tweak here:

- Sample length: how big of an audio clip should the encoder network process? 1 second of audio? 0.1 seconds of audio? 0.01 seconds of audio? To get an idea of how big each of these are, remember that 1 second of audio at 16khz is 16,000 int16 samples. 0.1 second clips are 1,600 samples. One MNIST image is 786 32 bit float values, so about the same overall size as 0.1 seconds of audio.
- Number of samples to sort: Should I train the network how to sort two samples of audio? That would mean just deciding which sample goes first and which goes second. Should I train it to sort three samples? How about five or twenty samples? Is there a sweet-spot?
- Padding size: How much padding should I put between each sample to ensure the network doesn’t cheat by simply matching up edges of samples that are closest in value. Is 0.01 seconds enough?
- Encoded representation size: How many features should we use for the encoded representation of each audio sample? In my previous work with MNIST I decided to use two features so that I could plot them on a two dimensional plot. For this though I’m not (yet) interested in plotting the samples in 2D space, so is 25 features good? How about 50 or 100?

If I had more intuition and experience working with audio I would probably have a good guess about where to start for each of these variables. But sadly I don’t. My solution: Experiment!

##### A link to my experiment ipynb is right [here](https://drive.google.com/file/d/0B_CaIm7pjWpfeTlJeUd2M25TRWc/view?usp=sharing).

So I set up my model to run in dozens of different configuration of each of these variables. That way I could look and see which combinations produce worthwhile results and which produce garbage. In this experiment phase I chose a very simple neural network model. For the encoder network, I used just two fully connected layers with 1000 neurons each followed by a fully connected layer with the number of neurons being the encoded features. 

The sorter network takes in *n* encoded samples of audio, where *n* is the number of samples we are sorting. Then it runs those through two fully connected layers with 1000 neurons each, followed by one layer with 200 neurons and the output of *nxn* softmax layers which is the prediction about the ordering.

I applied batch normalization to most of the layers to assist with training, as all together this network has about seven layers, which is pretty deep. I chose to start with very simple fully connected layers so I could focus on working out kinks with my algorithmic approach rather than focusing on making the most optimized and best model from the get-go. After getting things working okay with the fully connected layers I was going to try out fancier architectures, possibly with 1D convolutions and residual layers. If my experience in machine learning has taught me one thing, it’s that it’s usually best to start out simple and work your way slowly to more complex things. There are so many things that can go wrong and that are hard to debug, so it’s best to start simple and validate before optimizing.

## Validation and Problems I Faced

I ran my experimental configurations and got some pretty reassuring results. It seemed as though the network was learning how to un-shuffle the audio clips. As I kind of expected, performance was much better for sorting two or three samples, than much higher numbers. The network was usually also better at sorting smaller samples of audio than bigger samples.

Things looked great, but I still really couldn’t figure out *what* exactly my network was learning. In the talk I mentioned above where they sorted squares in a picture, they discussed how the network would frequently take shortcuts and ‘cheat.’ For example without padding it would just match up pixel colors along the edges of the squares and at one point it was even able to sort squares of seemingly random textures because it turns out the optics inside of cameras tends to make the top or bottom part of the image imperceptibly tinted red. The network was able to pick up that detail though.

So I thought a bit about how exactly I could detect how/when my network was cheating. The idea I came up with was to feed the network with samples from audio files that contained static noise, and audio files that contained a constant tone (so just a sin wave). I figured if the network figured out how to sort either of those then it somehow figured out a way to cheat.

I first plugged in two hours of [static noise](/assets/tf_4/static.mp3). Thankfully the network was not able to sort clips of this! When I told it to sort three samples it had an accuracy of ~0.333, which is just what I expected from a random guess. Not being able to sort this was a good sign because it meant that I wasn’t passing in any ‘hints’ to the network about how the samples were sorted.

Then I created a two hour audio file of a 440hz tone. When you listen to that it sounds like a flat tone, sort of like a beep. You can listen to it [here](/assets/tf_4/tone.mp3), just note that it can be pretty loud! I used this 440hz sin wave dataset to test that my model wouldn’t be able to piece together a steady tone. When you think about a sin wave:

 ![sin wave](/assets/tf_4/sin_wave.png)

You have to realize that it is a repeating pattern, so it would likely be difficult for anything or anyone to sort audio clips of a constant tone.

Unfortunately it seemed as though my network was actually learning how to sort the audio! That meant it was likely learning how to cheat somehow. I spent a great deal of time pinpointing exactly what was going on and why it was able to train. I feel like I learned a bunch from tracking down this problem actually.

The first problem I figured out was that I wasn’t actually changing the padding size used for each sample (remember the padding is used to stop the model from just matching up the values at the beginning and end of each sample). I had declared a variable in the global name space and I thought I reassigned it but instead of reassigning `pad_length` as I had intended, I was reassigning `pad_lengths`. So my program wasn’t crashing, but it was simply using a padding length of 0.001 regardless of what I wanted it to be. Moral of the story here is to be very careful with using the sandbox-like global space in ipython environments like Jupyter.

Another problem was that I noticed the network was likely able to extrapolate the tone’s sin function between samples when I used a constant padding size. So I decided to multiply my padding by a random number. When I first implemented this I made the random padding length the same for each sample. So if I was sorting a clip with three samples, there might be 0.01423 seconds of padding between each of those samples. What I realized though was that when the model was sorting more than two samples of audio, it could actually figure out the padding size between the samples. I was pretty fascinated when I realized that was what the network was doing. It kind of worries me that neural networks really do tend to take shortcuts wherever possible.

So that problem didn’t exist when I was sorting two audio samples. There was no way for it to predict the padding size with only one gap. But when I trained it on sorting 3+ samples, it was able to figure out on a per-sample basis how big the padding was, and extrapolate the sin wav to connect each sample. My solution here was to use random padding between each sample. So now if I’m sorting three samples of audio, the first two samples might have 0.0142 seconds of padding and the second two samples might have 0.00816 seconds of padding between them. After fixing this the network didn’t seem to be able to sort samples of the sin waves.

I had another hypothesis that perhaps there were periodic artifacts in how Audacity generated the sin wave audio. As in when Audacity discretized the sin wave by sampling it at 16bits by 16khz there was potentially a larger imperceptible pattern emerging. My workaround for this was to generate the audio with numpy by creating 1 cycle of the 440hz sin wav and copy/pasting that to make hours of audio in wav format. However I didn’t notice too much difference with this. So I don’t think it actually ended up being the problem.

A final little snag I ran into was memory management in Tensorflow. In my experiment I recreated every part of the network for every configuration I was testing. At first it would crash, running out of memory, after training a few configurations. It took a while of learning what Tensorflow does under the hood to solve this problem. My solution was to perform all of my work within a `with tf.Session() as sess:` block and then to reset the graph after the session had been cleared using `tf.reset_default_graph()`. I need to last part because both operations and variables take up precious GPU memory.

## What I’m working on next

I’ve shown now that I created a network that is somewhat able to learn how to sort samples of real music, but correctly fails at sorting samples of noise/tone. I honestly am still not entirely sure that the network isn’t cheating somehow, but I’m much more confident that it isn’t after testing it on the noise/tone.

Next I want to improve the performance of how the network. I’ve actually already begun doing this by using a deeper neural network architecture with 1 dimensional convolutional layers to process the samples. So far results seem promising!

I also want to be able to verify that my network is learning something about the underlying audio. I think the best way to verify this would be to chop off the sample encoding part of the model and either use it in some audio processing benchmark or possibly even using it in some sort of a generative model. I think I might try to perform deep dream-style generation using it. If anyone has any ideas on what I can use this model for, please feel free to share with me at ti@vt.edu! Or just share any thoughts on my approach, my writing in this post, or anything else!
