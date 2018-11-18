"""
xml2mask.py
Generate masks from xml files, which are from LabelMe http://labelme.csail.mit.edu/
Zhiang Chen, Nov 2018
Harish Hanand

Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
"""


import os
from lxml import etree
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pickle
import scipy.misc
import sys
from tqdm import tqdm

class polygonReader(object):
    def __init__(self, dataset, objects):
        """
        :param dataset: the name of the dataset, string
        :param objects: the names of the objects, a list of string
        """
        self.objects = objects
        self.path = "./datasets/" + dataset + "/annotation/"
        self.data = self.__getData__(objects)

    def __getData__(self, objects):
        files = [f for f in os.listdir(self.path)]
        data = dict()
        for file in files:
            if ".xml" in file:
                f = open(self.path + file, 'r')
                doc = etree.parse(f)
                data[file] = []
                if not bool(doc.xpath('/annotation/object')):
                    data[file] = None
                else:
                    for record in doc.xpath('/annotation/object'):
                        obj = record.xpath("name")[0].text
                        if obj in objects:
                            if record.xpath("deleted")[0].text == "0":
                                p = {obj:[]}
                                data[file].append(p)
                                polygon = []
                                for pt in record.xpath("polygon/pt"):
                                    polygon.append((int(pt.xpath("x")[0].text), int(pt.xpath("y")[0].text)))
                                i = len(data[file]) - 1
                                data[file][i][obj].append(tuple(polygon))
        return data

    def generateMask(self, dim=(400, 400)):
        """
        Generate all masks in one layer
        :param dim: the size of the mask, tuple
        :return: masks, ndarray
        """
        width, height = dim

        masks = dict()
        nm = len(self.objects)
        lines = dict()
        for i, obj in enumerate(self.objects):
            lines[obj] = (i+1)*255/nm

        for f, objects in self.data.iteritems():
            img = Image.new('L', (width, height), 0)
            if objects == None:
                masks[f] = np.array(img)
            else:
                for obj in objects:
                    name = obj.keys()[0]
                    l = lines[name]
                    poly = obj[name][0]
                    ImageDraw.Draw(img).polygon(poly, outline=l, fill=l)
                masks[f] = np.array(img)

        return masks

    def saveMask(self, dim=(400, 400)):
        masks = self.generateMask(dim)

        for file, mask in tqdm(masks.iteritems()):
            plt.imsave(self.path + file.split('.')[0]+'.jpg', mask, cmap="gray")


    def generateMask2(self, (width, height)=(400,400), resize=(400,400)):
        mask_dict = dict()
        for obj, polygons in self.data.iteritems():
            masks = np.zeros((len(polygons), resize[0], resize[1]))
            for i,poly in enumerate(polygons):
                img = Image.new('L', (width, height), 0)
                ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
                mask = np.array(img)*255
                mask = scipy.misc.imresize(mask, resize)
                masks[i,:,:] = mask
            mask_dict[obj] = masks # for visualization purpose
        return mask_dict


if __name__ == "__main__":
    objects = ['ndr', 'dr']
    polygon = polygonReader("tornado", objects)
    polygon.saveMask( dim=(4000, 4000) )
