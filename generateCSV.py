"""
generateCSV.py
Zhiang Chen, Jan 2019

Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

generate CSV files from npy files for RetinaNet repo: https://github.com/DREAMS-lab/keras-retinanet
"""

import numpy as np
import csv
import os

class generatorCSV(object):
    def __init__(self, npy_path, image_path, objects):
        self.npy_path = npy_path
        self.image_path = image_path
        self.objects = objects

        self.npy_files = [f for f in os.listdir(npy_path) if f.endswith(".npy")]
        self.__generate_csv__()

    def __generate_csv__(self):
        self.annotations = []
        for f in self.npy_files:
            image_name = f.split('.npy')[0] + '.png'
            masks = np.load(self.npy_path + f)
            dim = masks.shape
            if len(dim) == 2:
                annotation = [self.image_path + image_name, '0', '0', '0', '0', 'bg']
                self.annotations.append(annotation)
            else:
                for i in range(dim[-1]):
                    mask = masks[:,:,i]
                    bbox = self.__bbox__(mask)
                    if bbox is not None:
                        annotation = [self.image_path + image_name] + bbox
                        annotation.append(self.objects[0])
                        self.annotations.append(annotation)


    def __bbox__(self, mask):
        mask = np.transpose(mask)
        mask = np.where(mask != 0)
        if mask[0].size == 0:
            return None
        else:
            if np.min(mask[0]) == np.max(mask[0]):
                return None
            if np.min(mask[1]) == np.max(mask[1]):
                return None
            return [str(np.min(mask[0])), str(np.min(mask[1])), str(np.max(mask[0])), str(np.max(mask[1]))]

    def saveAnnotationCSV(self):
        with open('annotations.csv', mode='w') as csv_file:
            annotations_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for annotation in self.annotations:
                annotations_writer.writerow(annotation)

    def saveClassCSV(self):
        with open('classes.csv', mode='w') as csv_file:
            classes_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i, clas in enumerate(self.objects):
                classes_writer.writerow([clas,str(i)])


if __name__  ==  "__main__":
    npy_path = "../datasets/rocks_aug/val/"
    image_path = "../datasets/rocks_aug/val/"
    objects = ["rock"]
    csv_g = generatorCSV(npy_path, image_path, objects)
    csv_g.saveAnnotationCSV()
    csv_g.saveClassCSV()
