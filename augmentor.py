"""
augmentor.py
data augmentation
Zhiang Chen, Nov 2018

Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
"""
from scipy import ndimage
import numpy as np
import os

def sample(image, mask, rotation_min, rotation_max, fliplr, flipud, shift, zoom_min, zoom_max):

    return image, mask

def augmentor(image_path, annotation_path, mode=1, batch_number=1, rotation_min=0, rotation_max=0,
              fliplr=False, flipud=False, shift=0, zoom_min=1, zoom_max=1):
    """
    :param image_path:
    :param annotation_path:
    :param mode: 1: image is RGB, annotation is grayscale; 2: image is RGB, annotation is ndarray (.npy)
    :param batch_number:
    :param rotation_min:
    :param rotation_max:
    :param fliplr:
    :param flipud:
    :param shift:
    :param zoom_min:
    :param zoom_max:
    :return:
    """
    c = 0
    image_files = [f for f in os.listdir(image_path)]
    annotation_files = [f for f in os.listdir(annotation_path)]
    annotation_prefix = [f.split('.')[0] for f in annotation_files]

    while c < batch_number:
        c += 1
        for image in image_files:
            if image.split(".")[0] in annotation_prefix:
                if mode == 1:
                    pass
                elif mode == 2:
                    pass
                else:
                    print("do something to process this scenario")


                img,mask = sample(img, mask, rotation_min, rotation_max,
                                  fliplr, flipud, shift, zoom_min, zoom_max)
                yield img,mask

            else:
                print("Cannot find the corresponding anntation file for " + image)




if __name__ == "__main__":
    config = dict(
                batch_number = 10,
                rotation_min = -90,
                rotation_max = 90,
                fliplr = True,
                flipud = True,
                shift = 0.2,
                zoom_min = 0.8,
                zoom_max = 1.2)

    image_path = './datasets/tornado/img/'
    annotation_path = './datasets/tornado/ann/'
    aug = augmentor(image_path, annotation_path, **config)

    for c in aug:
        print(c)

