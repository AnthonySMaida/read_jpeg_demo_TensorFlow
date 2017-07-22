#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:40:09 2017

@author: maida

Read jpeg, convert to grayscale, resize, and normalize image
only using TensorFlow APIs.
Uses matplotlib to display image.
Prints image properties of unevaluated and evaluated tensors.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IM_SZ_LEN = 64 # For later experiments, increase size as necessary
IM_SZ_WID = 64

model = tf.Graph()
with model.as_default():
    file_contents = tf.read_file('image_0004_leafCropped.jpg')
    image         = tf.image.decode_jpeg(file_contents)
    image         = tf.image.rgb_to_grayscale(image) # Input to the LSTM !!!
    image         = tf.image.resize_images(image, [IM_SZ_LEN, IM_SZ_WID])
    image         = tf.expand_dims(image, 0)         # creates tensor of rank 4
    image         = (1/255.0) * image                # normalize to range 0-1
    print("Properties of uneval'd image tensor")
    print("   Shape of image: ", tf.shape(image))
    print("   Rank of  image: ", tf.rank(image))
    print("   Size of  image: ", tf.size(image))

with tf.Session(graph=model) as sess:
    print("Properties of eval'd image tensor")
    print("   Shape of image: ", tf.shape(image).eval())
    print("   Rank of  image: ", tf.rank(image).eval())
    print("   Size of  image: ", tf.size(image).eval())
    output = sess.run(image)
    

print('Output shape after run() evaluation: ', output.shape)
output.resize((IM_SZ_LEN, IM_SZ_WID))
print('Resized for plt.imshow() by removing null dimensions:', output.shape)
print('Print some matrix values to show it is grayscale.')
print(output)
print('Display the grayscale image.')
plt.imshow(output, cmap = cm.Greys_r)
