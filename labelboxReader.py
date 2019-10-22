"""
Zhiang Chen, Oct 21, 2019
"""

import numpy as np
import json


class LabelboxReader(object):
    def __init__(self, image_size=(W, H)):
        self.W, self.H = image_size
        self.labels_dict = {}

    def readJson(self, path):
        with open(path, 'r') as f:
            self.labels_dict = json.load(f)

    def convert2ndarray(self, mask_lists):
        cls = 0
        nd = 0
        return cls, nd

    def convert(self):
        for label in self.labels_dict:
            dataset_name = label['Dataset Name']
            image_name = label['External ID']

            if label['Label'] == 'Skip':
                nd_label = np.zeros((self.W, self.H, 1))
                cls = np.zeros(1)
            elif label['Label'].get('tile_dmg', False):
                if label['Label']['tile_dmg'] == 'nd':
                    nd_label = np.zeros((self.W, self.H, 1))
                    cls = np.zeros(1)
                else:
                    cls, nd_label = self.convert2ndarray(label['Label'])
            else:
                cls, nd_label = self.convert2ndarray(label['Label'])

            # save ndarray and classes in two files

if __name__  ==  "__main__":
    lb = LabelboxReader(image_size=(400, 400))
    lb.readJson("../eureka_labels.json")
    lb.convert()