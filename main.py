import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageDraw
import face_recognition

from glob import glob

import IPython.display as ipd
from tqdm import tqdm

import subprocess

# Load in video capture
cap = cv2.VideoCapture('world.mp4')
num_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Video height and width
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# num of frames per second
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'FPS : {fps:0.2f}')

# pull image from video
ret, img = cap.read()
print(f'Returned {ret} and img of shape {img.shape}')


# show multiple frames from video
def multiple_frames():

    fig, axs = plt.subplots(10, 10, figsize=(30, 20))
    axs = axs.flatten()

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    img_idx = 0
    for frame in range(n_frames):
        ret, img = cap.read()
        if ret == False:
            break
        if frame in range(100):    # show first 100 hundred frames
            axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[img_idx].set_title(f'Frame: {frame}')
            axs[img_idx].axis('off')
            img_idx += 1

    plt.tight_layout()
    plt.show()


# show one frame
def display_cv2_img(img, figsize=(10, 10)):
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_)
    ax.axis("off")

    plt.tight_layout()
    plt.show()

def get_one_frame():

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in range(n_frames):
        ret, img = cap.read()
        if ret == False:
            break
        if frame == 150: # 150, 210, 450, 900   # get one frame, number 600 for example
            break

    faces = face_recognition.face_locations(img)
    face_locations = face_recognition.face_locations(img, model="cnn")
    print(face_locations)
    x1 = face_locations[0][3]
    y1 = face_locations[0][2]

    x2 = face_locations[0][1]
    y2 = face_locations[0][0]

    pt1 = (x1, y1)
    pt2 = (x2, y2)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)

    display_cv2_img(img)


get_one_frame()

