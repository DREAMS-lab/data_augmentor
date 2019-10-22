"""
Zhiang Chen, Oct 21, 2019
"""

import numpy as np
import json
from PIL import Image, ImageDraw


class LabelboxReader(object):
    def __init__(self, image_size):
        self.H, self.W = image_size
        self.labels_dict = {}

    def readJson(self, path):
        with open(path, 'r') as f:
            self.labels_dict = json.load(f)

    def draw_poly(self, points_list):
        polygon = []
        for point in points_list:
            x = point['x']
            y = point['y']
            polygon.append((x, y))
        if len(polygon) > 2:
            img = Image.new('L', (self.H, self.W), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=255, fill=255)
        return polygon

    def convert2ndarray(self, masks_dict):
        cls = []
        nd = []
        if masks_dict.get('nd_roof', False):
            for nd_roof in masks_dict['nd_roof']:
                cls.append(0)
                nd.append(self.draw_poly(nd_roof['geometry']))
        if masks_dict.get('d0_roof', False):
            for d0_roof in masks_dict['d0_roof']:
                cls.append(1)
                nd.append(self.draw_poly(d0_roof['geometry']))
        if masks_dict.get('d1_roof', False):
            for d1_roof in masks_dict['d1_roof']:
                cls.append(2)
                nd.append(self.draw_poly(d1_roof['geometry']))
        if masks_dict.get('d2_roof', False):
            for d2_roof in masks_dict['d2_roof']:
                cls.append(3)
                nd.append(self.draw_poly(d2_roof['geometry']))
        if masks_dict.get('d3_roof', False):
            for d3_roof in masks_dict['d3_roof']:
                cls.append(4)
                nd.append(self.draw_poly(d3_roof['geometry']))
        return cls, nd

    def convert(self):
        for label in self.labels_dict:
            dataset_name = label['Dataset Name']
            image_name = label['External ID']

            if label['Label'] == 'Skip':
                nd_label = np.zeros((self.H, self.W, 1))
                cls = np.zeros(1)
            elif label['Label'].get('tile_dmg', False):
                if label['Label']['tile_dmg'] == 'nd':
                    nd_label = np.zeros((self.H, self.W, 1))
                    cls = np.zeros(1)
                else:
                    cls, nd_label = self.convert2ndarray(label['Label'])
            else:

                cls, nd_label = self.convert2ndarray(label['Label'])

            # save ndarray and classes in two files
            cls_file_name = dataset_name + "_" + image_name.split('.')[0] + "_cls.npy"
            nd_file_name = dataset_name + "_" + image_name.split('.')[0] + "_nd.npy"
            np.save("../labels/" + cls_file_name, cls)
            np.save("../labels/" + nd_file_name, nd_label)

if __name__  ==  "__main__":
    lb = LabelboxReader(image_size=(2000, 2000))
    lb.readJson("../eureka_labels.json")
    lb.convert()