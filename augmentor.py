"""
augmentor.py
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
  image_center = tuple(np.array(image.shape[:2]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
  return result

def view_channel(image, c=0):
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

def sample(image, mask, rotation_min, rotation_max, fliplr, flipud, zoom_min, zoom_max):
    print(image.shape)
    print(mask.shape)
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

    image = cv2.resize(image, None, fx=zoom_scale, fy=zoom_scale)
    mask = cv2.resize(mask, None, fx=zoom_scale, fy=zoom_scale)

    if zoom_scale>1:
        pass
    elif zoom_scale == 1:
        pass
    else:
        pass

    return image, mask

def augmentor(image_path, annotation_path, mode=1, batch_number=1, rotation_min=0, rotation_max=0,
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
    :param shift:
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
        c += 1
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

                print(annotation)
                img,mask = sample(img, mask, rotation_min, rotation_max,
                                  fliplr, flipud, zoom_min, zoom_max)
                yield img,mask

            else:
                print("Cannot find the corresponding anntation file for " + image)

'''
def resize(image, new_size, keep_size=True):
    """
    resize the multi-channel image.
    :param image: multi-channel image, ndarray, shape=[width, height, channel]
    :param size: new size, tuple
    :param keep_size: keep the original size if True, then fill the boundaries with 0 if needed
    :return: resized image, ndarray
    Note this only support the intensity ranging (0,255)
    """
    size = image.shape
    if size[0:2] == new_size:
        return image

    if keep_size:
        # either greater or smaller on width and height dimensions
        assert (new_size[0] - size[0]) * (new_size[1] - size[1]) >= 0

    if len(image.shape) == 2:
        image = imresize(image, new_size, mode='RGB')
        if keep_size:
            if ((new_size[0] - size[0]) >= 0) & ((new_size[1] - size[1]) >= 0):
                image = image[(new_size[0]-size[0])/2:(new_size[0]-size[0])/2+size[0],
                        (new_size[1]-size[1])/2:(new_size[1]-size[1])/2+size[1]]
            else:
                new_image = np.zeros(size)
                new_image[(size[0]-new_size[0])/2:(size[0]-new_size[0])/2+new_size[0],
                (size[1]-new_size[1])/2:(size[1]-new_size[1])/2+new_size[1]] = image
                image = new_image

    else:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 1).tolist()
        image = np.array([imresize(np.array(img, mode='RGB'), new_size) for img in image])
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)

        if keep_size:
            if ((new_size[0] - size[0]) >= 0) & ((new_size[1] - size[1]) >= 0):
                image = image[(new_size[0] - size[0]) / 2:(new_size[0] - size[0]) / 2 + size[0],
                        (new_size[1] - size[1]) / 2:(new_size[1] - size[1]) / 2 + size[1], :]
            else:
                new_image = np.zeros(size)
                new_image[(size[0] - new_size[0]) / 2:(size[0] - new_size[0]) / 2 + new_size[0],
                (size[1] - new_size[1]) / 2:(size[1] - new_size[1]) / 2 + new_size[1], :] = image
                image = new_image

    return image
'''

if __name__  ==  "__main__":
    config = dict(
                mode=2,
                batch_number=2,
                rotation_min=-90,
                rotation_max=90,
                fliplr=True,
                flipud=True,
                zoom_min=0.8,
                zoom_max=1.2)

    image_path = './datasets/tornado/img/'
    annotation_path = './datasets/tornado/ann/'
    aug = augmentor(image_path, annotation_path, **config)

    for i,m in aug:
        #print(i.shape)
        #print(m.shape)
        print('...')

