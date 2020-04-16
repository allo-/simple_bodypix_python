#!/usr/bin/env python3
import tensorflow as tf
import cv2
import sys
from PIL import Image
import tfjs_graph_converter as tfjs
import numpy as np
import os
import stat
import glob
import yaml
import pyfakewebcam

replacement_bgs = None
replacement_bgs_idx = 0
replacement_bgs_mtime = 0
config_mtime = 0
config = {
    "erode": 0,
    "blur": 0,
    "segmentation_threshold": 0.75,
    "blur_background": 0,
    "image_name": "background.jpg",
    "virtual_video_device": "/dev/video2",
    "real_video_device": "/dev/video0",
    "average_masks": 3
}
masks = []

def load_config():
    global config_mtime, config, replacement_bgs_mtime
    try:
        if os.stat("config.yaml").st_mtime != config_mtime:
            config_mtime = os.stat("config.yaml").st_mtime
            with open("config.yaml", "r") as configfile:
                yconfig = yaml.load(configfile, Loader=yaml.SafeLoader)
                for key in yconfig:
                    config[key] = yconfig[key]
            # Force image reload
            replacement_bgs_mtime = 0
    except OSError:
        pass
    return config

load_config()

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# make tensorflow stop spamming messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
tf.get_logger().setLevel("DEBUG")

cap = cv2.VideoCapture(config.get("real_video_device"))

# configure camera for 720p @ 30 FPS
height, width = 720, 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH ,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv2.CAP_PROP_FPS, 30)

fakewebcam = pyfakewebcam.FakeWebcam(config.get("virtual_video_device"), width, height)

# CONSTANTS
OutputStride = 16
internalResolution = 0.5

# PATHS
modelPath = 'bodypix_mobilenet_float_{0:03d}_model-stride{1}'.format(
    int(100*internalResolution), OutputStride)

#tf.debugging.set_log_device_placement(True)

print("Loading model...")
graph = tfjs.api.load_graph_model(modelPath)  # downloaded from the link above
print("done.")

def load_replacement_bgs(replacement_bgs, image_name="background.jpg", blur_background_value=0):
    global replacement_bgs_mtime, replacement_bgs_idx
    try:
        replacement_stat = os.stat(image_name)
        if replacement_stat.st_mtime != replacement_bgs_mtime:
            print("Loading background {0} ...".format(image_name))
            replacement_bgs_idx = 0
            filenames = [image_name]
            if stat.S_ISDIR(replacement_stat.st_mode):
                filenames = glob.glob(filenames[0] + "/*.*")
                if not filenames:
                    return None

            replacement_bgs = []
            for filename in filenames:
                replacement_bg_raw = cv2.imread(filename)
                interpolation_method = cv2.INTER_LINEAR
                if config.get("background_interpolation_method") == "NEAREST":
                    interpolation_method = cv2.INTER_NEAREST
                replacement_bg = cv2.resize(replacement_bg_raw, (width, height),
                    interpolation=interpolation_method)
                replacement_bg = replacement_bg[...,::-1]
                replacement_bgs.append(replacement_bg)

            replacement_bgs_mtime = os.stat(image_name).st_mtime

            if blur_background_value:
                for i in range(len(replacement_bgs)):
                    replacement_bgs[i] = cv2.blur(replacement_bgs[i],
                        (blur_background_value, blur_background_value))
            print("Finished loading background")

        return replacement_bgs

    except OSError:
        return None

sess = tf.compat.v1.Session(graph=graph)
input_tensor_names = tfjs.util.get_input_tensors(graph)
output_tensor_names = tfjs.util.get_output_tensors(graph)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

def calc_padding(inputTensor, targetH, targetW):
    height, width = inputTensor.shape[:2]
    targetAspect = targetW / targetH;
    aspect = width / height;
    padT, padB, padL, padR = 0, 0, 0, 0;
    if aspect < targetAspect:
        padT = 0
        padB = 0
        padL = round(0.5 * (targetAspect * height - width))
        padR = round(0.5 * (targetAspect * height - width))
    else:
        padT = round(0.5 * ((1.0 / targetAspect) * width - height))
        padB = round(0.5 * ((1.0 / targetAspect) * width - height))
        padL = 0
        padR = 0
    return padT, padB, padL, padR

def removePaddingAndResizeBack(resizedAndPadded, originalHeight, originalWidth,
        padT, padB, padL, padR):
    return tf.squeeze(tf.image.crop_and_resize(resizedAndPadded,
        [[padT / (originalHeight + padT + padB - 1.0),
        padL / (originalWidth + padL + padR - 1.0),
        (padT + originalHeight - 1.0) / (originalHeight + padT + padB - 1.0),
        (padL + originalWidth - 1.0) / (originalWidth + padL + padR - 1.0)]],
        [0], [originalHeight, originalWidth]
    ), [0])

def scaleAndCropToInputTensorShape(tensor, inputTensorHeight, inputTensorWidth,
        resizedAndPaddedHeight, resizedAndPaddedWidth, padT, padB, padL, padR,
        applySigmoidActivation):
    inResizedAndPadded = tf.image.resize_with_pad(tensor,
        inputTensorHeight, inputTensorWidth,
        method=tf.image.ResizeMethod.BILINEAR)
    if applySigmoidActivation:
        inResizedAndPadded = tf.sigmoid(inResizedAndPadded)

    return removePaddingAndResizeBack(inResizedAndPadded,
        inputTensorHeight, inputTensorWidth, padT, padB, padL, padR)

def isValidInputResolution(resolution, outputStride):
    return (resolution - 1) % outputStride == 0;

def toValidInputResolution(inputResolution, outputStride):
    if isValidInputResolution(inputResolution, outputStride):
        return inputResolution
    return int(np.floor(inputResolution / outputStride) * outputStride + 1)

def toInputResolutionHeightAndWidth(internalResolution, outputStride, inputHeight, inputWidth):
    return (toValidInputResolution(inputHeight * internalResolution, outputStride),
            toValidInputResolution(inputWidth * internalResolution, outputStride))

def toMaskTensor(segmentScores, threshold):
    return tf.math.greater(scaledSegmentScores, tf.constant(threshold))

while True:
    config = load_config()
    success, frame = cap.read()
    if not success:
        print("Error getting a webcam image!")
        sys.exit(1)

    if config.get("flip_horizontal"):
        frame = cv2.flip(frame, 1)
    if config.get("flip_vertical"):
        frame = cv2.flip(frame, 0)

    blur_background_value = config.get("blur_background", 0)
    image_name = config.get("image_name", "background.jpg")
    replacement_bgs = load_replacement_bgs(replacement_bgs, image_name, blur_background_value)

    frame = frame[...,::-1]
    if replacement_bgs is None:
        if not blur_background_value:
            fakewebcam.schedule_frame(frame)
            continue
        elif blur_background_value:
            replacement_bgs = frame
            replacement_bgs = cv2.blur(replacement_bgs,
                (blur_background_value, blur_background_value))

    img = Image.fromarray(frame)
    imgWidth, imgHeight = img.size

    targetHeight, targetWidth = toInputResolutionHeightAndWidth(
        internalResolution, OutputStride, imgHeight, imgWidth)


    x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
    padT, padB, padL, padR = calc_padding(x, targetHeight, targetWidth)
    x = tf.image.resize_with_pad(x, targetHeight, targetWidth,
            method=tf.image.ResizeMethod.BILINEAR)

    resizedHeight, resizedWidth = x.shape[:2]

    InputImageShape = x.shape

    widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
    heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1

    # for resnet
    # add imagenet mean - extracted from body-pix source
    #m = np.array([-123.15, -115.90, -103.06])
    #x = np.add(x, m)

    # for mobilenet
    x = np.divide(x, 127.5)
    x = np.subtract(x, 1.0)
    sample_image = x[tf.newaxis, ...]

    results = sess.run(output_tensor_names, feed_dict={
                       input_tensor: sample_image})
    segments = np.squeeze(results[1], 0)

    segmentLogits = results[1]
    scaledSegmentScores = scaleAndCropToInputTensorShape(
        segmentLogits, imgHeight, imgWidth, resizedHeight, resizedWidth,
        padT, padB, padL, padR, True
    )

    mask = toMaskTensor(scaledSegmentScores, config["segmentation_threshold"])
    segmentationMask = tf.dtypes.cast(mask, tf.int32)
    segmentationMask = np.reshape(
        segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))

    masks.insert(0, segmentationMask)
    num_average_masks = max(1, config.get("average_masks", 3))
    masks = masks[:num_average_masks]
    segmentationMask = np.mean(masks, axis=0)

    mask_img = Image.fromarray(segmentationMask * 255).convert("RGB")
    #DEBUG
    #mask_img = tf.keras.preprocessing.image.img_to_array(
    #    mask_img, dtype=np.uint8)
    #frame = np.array(mask_img[:,:,:])
    #fakewebcam.schedule_frame(frame)
    #cv2.imwrite("output.jpg", frame)
    #break

    mask_img = tf.keras.preprocessing.image.img_to_array(
        mask_img, dtype=np.uint8)

    if config["dilate"]:
        mask_img = cv2.dilate(mask_img, np.ones((config["dilate"], config["dilate"]), np.uint8), iterations=1)
    if config["erode"]:
        mask_img = cv2.erode(mask_img, np.ones((config["erode"], config["erode"]), np.uint8), iterations=1)
    if config["blur"]:
        mask_img = cv2.blur(mask_img, (config["blur"], config["blur"]))
    segmentationMask_inv = np.bitwise_not(mask_img)

    for c in range(3):
        frame[:,:,c] = frame[:,:,c] * (mask_img[:,:,0] / 255.) + \
            replacement_bgs[replacement_bgs_idx][:,:,c] * (1.0-(mask_img[:,:,0] / 255.))

    replacement_bgs_idx = (replacement_bgs_idx + 1) % len(replacement_bgs)

    if config.get("debug_show_mask", False):
        frame = np.array(mask_img[:,:,:])
    fakewebcam.schedule_frame(frame)

sys.exit(0)
