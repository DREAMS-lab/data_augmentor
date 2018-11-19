"""
rename.py
rename training and annotation files by numerical order
Zhiang Chen, Nov 2018

Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
"""


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

class renamer(object):
    def __init__(self, image_path, annotation_path):
        """
        :param image_path: the relative path of image
        :param annotation_path: the relative path of annotation
        """
        self.image_path = image_path
        self.annotation_path = annotation_path

        self.image_files = [f for f in os.listdir(image_path)]
        self.annotation_files = [f for f in os.listdir(annotation_path)]

    def rename(self, mode, image2png=False, image_color=True, annotation_suffix='.jpg', annotation2suffix='.png'):
        """
        :param mode: 1, numerical oder; 2, number order with prefix as 'label_'
        :return:
        """
        c = 0
        annotation_prefix = [f.split('.')[0] for f in self.annotation_files]

        for image in self.image_files:
            if image.split('.')[0] in annotation_prefix:

                if image_color:
                    im = Image.open(self.image_path + image)
                    if image2png:
                        im.save(self.image_path + str(c) + '.png')
                    else:
                        im.save(self.image_path + str(c) + '.jpg')
                else:
                    print("do something to process grayscale images")

                annotation = image.split('.')[0] + annotation_suffix
                if annotation_suffix == '.jpg':
                    if annotation2suffix == '.jpg':
                        if mode == 1:
                            os.rename(self.annotation_path + annotation, self.annotation_path + str(c) + '.jpg')
                        elif mode == 2:
                            os.rename(self.annotation_path + annotation, self.annotation_path + "label_" + str(c) + '.jpg')
                    elif annotation2suffix == '.png':
                        if mode == 1:
                            im = cv2.imread(self.annotation_path + annotation, 0)
                            cv2.imwrite(self.annotation_path + str(c) + '.png', im)
                        elif mode == 2:
                            im = cv2.imread(self.annotation_path + annotation, 0)
                            cv2.imwrite(self.annotation_path + "label_" + str(c) + '.png', im)
                    else:
                        print("do something to process this")

                elif annotation_suffix == '.png':
                    if annotation2suffix == '.png':
                        if mode == 1:
                            os.rename(self.annotation_path + annotation, self.annotation_path + str(c) + '.png')
                        elif mode == 2:
                            os.rename(self.annotation_path + annotation, self.annotation_path + "label_" + str(c) + '.png')
                    else:
                        print("do something to process this")

                else:
                    os.rename(self.annotation_path + annotation, self.annotation_path + str(c) + annotation2suffix)

                c += 1
            else:
                print("Cannot find corresponding annotation file for "+image)


if __name__ == "__main__":
    rn = renamer('./datasets/tornado/img/', './datasets/tornado/ann/')
    rn.rename(mode=1, annotation_suffix='.jpg', annotation2suffix='.png')