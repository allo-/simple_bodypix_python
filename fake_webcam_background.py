#!/usr/bin/env python3
import cv2
import sys
from PIL import Image
import tfjs_graph_converter as tfjs
import tensorflow as tf
import math
import matplotlib.patches as patches
import numpy as np
import os
import pyfakewebcam

# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

cap = cv2.VideoCapture('/dev/video0')

# configure camera for 720p @ 30 FPS
height, width = 720, 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH ,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv2.CAP_PROP_FPS, 30)

fakewebcam = pyfakewebcam.FakeWebcam("/dev/video2", width, height)

# PATHS
modelPath = 'bodypix_resnet50_float_model-stride16'

# CONSTANTS
OutputStride = 16

#tf.debugging.set_log_device_placement(True)

print("Loading model...")
graph = tfjs.api.load_graph_model(modelPath)  # downloaded from the link above
print("done.")


replacement_bg_raw = cv2.imread("background.jpg")
replacement_bg = cv2.resize(replacement_bg_raw, (width, height))
replacement_bg = replacement_bg[...,::-1]

sess = tf.compat.v1.Session(graph=graph)
input_tensor_names = tfjs.util.get_input_tensors(graph)
output_tensor_names = tfjs.util.get_output_tensors(graph)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

while True:
    success, frame = cap.read()
    if not success:
        print("Error getting a webcam image!")
        sys.exit(1)

    frame = frame[...,::-1]

    img = Image.fromarray(frame)
    imgWidth, imgHeight = img.size

    targetWidth = (int(imgWidth) // OutputStride) * OutputStride + 1
    targetHeight = (int(imgHeight) // OutputStride) * OutputStride + 1

    img = img.resize((targetWidth, targetHeight))
    x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
    InputImageShape = x.shape

    widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
    heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1

    # add imagenet mean - extracted from body-pix source
    m = np.array([-123.15, -115.90, -103.06])
    x = np.add(x, m)
    sample_image = x[tf.newaxis, ...]

    results = sess.run(output_tensor_names, feed_dict={
                       input_tensor: sample_image})
    segments = np.squeeze(results[6], 0)

    # Segmentation MASK
    segmentation_threshold = 0.7
    segmentScores = tf.sigmoid(segments)
    mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
    segmentationMask = tf.dtypes.cast(mask, tf.int32)
    segmentationMask = np.reshape(
        segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))

    # Draw Segmented Output
    mask_img = Image.fromarray(segmentationMask * 255)
    mask_img = mask_img.resize(
        (targetWidth, targetHeight), Image.LANCZOS).convert("RGB")
    mask_img = tf.keras.preprocessing.image.img_to_array(
        mask_img, dtype=np.uint8)

    segmentationMask_inv = np.bitwise_not(mask_img)
    #fg = np.bitwise_and(np.array(img), np.array(
    #    mask_img))

    #frame = frame[...,::-1] # convert frame back to BGR
    frame = np.bitwise_and(frame, mask_img[:-1,:-1,:]) + \
            np.bitwise_and(replacement_bg, segmentationMask_inv[:-1,:-1,:])
    fakewebcam.schedule_frame(frame)

sys.exit(0)
