"""
augmentation.py
data augmentation
Zhiang Chen, Nov 2018

Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

Using cv2 to read images, which is faster: https://www.kaggle.com/zfturbo/test-speed-cv2-vs-scipy-vs-tensorflow
"""

#from scipy import ndimage # bad and slow
#from scipy.misc import imresize  # very bad function
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def rotateImage(image, angle):
    image = image.astype(np.uint8)
    l = len(image.shape)
    image_center = tuple(np.array(image.shape[:2]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
    if len(result.shape)<l:
        y,x = result.shape
        result = result.reshape((y,x,1))
    return result

def viewChannel(image, c=0):
    """
    visualize one channel of the multi-channel image
    :param image: multi-channel image, ndarray
    :param c: channel to look at, int
    :return:
    """
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
        plt.show()
    else:
        assert c < image.shape[2]
        img = image[:,:,c]
        plt.imshow(img, cmap='gray')
        plt.show()

def zoom(image, zoom_scale):
    size = image.shape
    l = len(size)
    image = cv2.resize(image, None, fx=zoom_scale, fy=zoom_scale)
    if len(image.shape) < l:
        y,x = image.shape
        image = image.reshape((y,x,1))
    new_size = image.shape
    
    if len(size) == 3:
        if zoom_scale > 1:
            return image[int((new_size[0] - size[0]) / 2):int((new_size[0] - size[0]) / 2 + size[0]),
                   int((new_size[1] - size[1]) / 2):int((new_size[1] - size[1]) / 2 + size[1]), :]
        elif zoom_scale == 1:
            return image
        else:
            new_image = np.zeros(size).astype('uint8')
            new_image[int((size[0] - new_size[0]) / 2):int((size[0] - new_size[0]) / 2 + new_size[0]),
            int((size[1] - new_size[1]) / 2):int((size[1] - new_size[1]) / 2 + new_size[1]), :] = image
            return new_image
    else:
        if zoom_scale > 1:
            return image[int((new_size[0]-size[0])/2):int((new_size[0]-size[0])/2+size[0]),
                        int((new_size[1]-size[1])/2):int((new_size[1]-size[1])/2+size[1])]
        elif zoom_scale == 1:
            return image
        else:
            new_image = np.zeros(size).astype('uint8')
            new_image[int((size[0] - new_size[0]) / 2):int((size[0] - new_size[0]) / 2 + new_size[0]),
            int((size[1] - new_size[1]) / 2):int((size[1] - new_size[1]) / 2 + new_size[1])] = image
            return new_image


def sample(image, mask, rotation_min, rotation_max, fliplr, flipud, zoom_min, zoom_max):
    angle = np.random.uniform(rotation_min, rotation_max)
    image = rotateImage(image, angle)
    mask = rotateImage(mask, angle)
    
    if fliplr:
        if np.random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    if flipud:
        if np.random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
            
    zoom_scale = np.random.uniform(zoom_min, zoom_max)
    image = zoom(image, zoom_scale)
    mask = zoom(mask, zoom_scale)
    
    return image, mask

def augmentor(image_path, annotation_path, mode=1, resize_dim=None, batch_number=1, rotation_min=0, rotation_max=0,
              fliplr=False, flipud=False, zoom_min=1, zoom_max=1):
    """
    :param image_path:
    :param annotation_path:
    :param mode: 1: image is RGB, annotation is grayscale; 2: image is RGB, annotation is ndarray (.npy)
    :param batch_number:
    :param rotation_min:
    :param rotation_max:
    :param fliplr:
    :param flipud:
    :param zoom_min:
    :param zoom_max:
    :return:
    """
    c = 0
    image_files = [f for f in os.listdir(image_path)]
    annotation_files = [f for f in os.listdir(annotation_path)]
    annotation_prefix = [f.split('.')[0] for f in annotation_files]
    annotation_suffix = '.' + annotation_files[0].split('.')[-1]

    while c < batch_number:
        for image in image_files:
            if image.split(".")[0] in annotation_prefix:
                annotation = image.split(".")[0] + annotation_suffix
                if mode == 1:  # image is RGB, annotation is grayscale
                    img = cv2.imread((image_path + image), cv2.IMREAD_UNCHANGED)
                    mask = cv2.imread((annotation_path + annotation), cv2.IMREAD_GRAYSCALE)

                elif mode == 2:  # image is RGB, annotation is ndarray (.npy)
                    img = cv2.imread((image_path + image), cv2.IMREAD_UNCHANGED)
                    mask = np.load(annotation_path + annotation)
                else:
                    print("do something to process this scenario")
                
                # bug: when the mask size is (y,x,1) the return is (y,x)
                # bug: resize also transpose the desired size
                l = len(mask.shape)
                if resize_dim != None:
                    img = cv2.resize(img, dsize=resize_dim)
                    mask = cv2.resize(mask, dsize=resize_dim)
                    if len(mask.shape) < l:
                        y,x = mask.shape
                        mask = mask.reshape((y, x, 1))
                

                img,mask = sample(img, mask, rotation_min, rotation_max,
                                  fliplr, flipud, zoom_min, zoom_max)
                yield img, mask, image.split('.')[0]

            else:
                print("Cannot find the corresponding anntation file for " + image)
        c += 1

if __name__  ==  "__main__":
    config = dict(
                mode=2,
                resize_dim=(500, 500),
                batch_number=2,
                rotation_min=-90,
                rotation_max=90,
                fliplr=True,
                flipud=True,
                zoom_min=0.8,
                zoom_max=1.2)

    image_path = './datasets/Crater/image/'
    annotation_path = './datasets/Crater/npy/'
    aug = augmentor(image_path, annotation_path, **config)

    for i,m,f in aug:
        print('...')

