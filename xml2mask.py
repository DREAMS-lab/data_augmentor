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
import pickle
import scipy.misc
from tqdm import tqdm
import cv2



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
                                if len(polygon) == 0:
                                    xmin = int(record.xpath("segm/box/xmin")[0].text)
                                    xmax = int(record.xpath("segm/box/xmax")[0].text)
                                    ymin = int(record.xpath("segm/box/ymin")[0].text)
                                    ymax = int(record.xpath("segm/box/ymax")[0].text)
                                    polygon.append((xmin, ymin))
                                    polygon.append((xmin, ymax))
                                    polygon.append((xmax, ymax))
                                    polygon.append((xmax, ymin))
                                i = len(data[file]) - 1
                                data[file][i][obj].append(tuple(polygon))
        return data

    def generateMask(self, dim=(400, 400)):
        """
        Generate all masks in one layer
        :param dim: the size of the mask, tuple
        :return: masks, {file1: ndarray(mask1), file2: ndarray(mask2), ...}
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
            cv2.imwrite(self.path + file.split('.')[0]+'.jpg', mask)


    def generateMask2(self, dim=(400,400), resize=(400,400), saveOnline=False):
        """
        Generate masks on individual layers, the classes of objects is also represented by intensities ranging from (0, 255)
        :param dim: original dimension of masks, tuple
        :param resize: resize dimension of masks, tuple
        :return: multilayer Masks, {file1:ndarray(mask1), file2:ndarray(mask2), ...} if not saveOnline; otherwise, save masks, and return is meaningless
        Note: the .npy is of dimension (width, height, nm_objects)
        """

        width, height = dim
        masks = dict()
        nm = len(self.objects)
        lines = dict()
        for i, obj in enumerate(self.objects):
            lines[obj] = (i+1)*255/nm

        #for f, objects in tqdm(self.data.iteritems()):  # python2
        for f, objects in tqdm(self.data.items()):  # python3
            if objects == None:
                img = Image.new('L', (resize[0], resize[1]), 0)
                masks[f] = np.array(img)
                if saveOnline:
                    np.save(self.path + f.split('.')[0], np.array(img))
                    masks[f] = None
            else:
                masks_ndarray = np.zeros((len(objects), resize[0], resize[1]))
                for i, obj in enumerate(objects):
                    img = Image.new('L', (width, height), 0)
                    name = obj.keys()[0]
                    l = lines[name]
                    poly = obj[name][0]
                    ImageDraw.Draw(img).polygon(poly, outline=l, fill=l)
                    mask = np.array(img)
                    mask = scipy.misc.imresize(mask, resize)
                    masks_ndarray[i, :, :] = mask
                masks_ndarray = np.swapaxes(masks_ndarray, 0, 1)
                masks_ndarray = np.swapaxes(masks_ndarray, 1, 2)
                if not saveOnline:
                    masks[f] = masks_ndarray
                else:
                    np.save(self.path + f.split('.')[0], masks_ndarray)
                    masks[f] = None




if __name__ == "__main__":
    objects = ['ndr', 'dr']
    polygon = polygonReader("tornado", objects)
    polygon.saveMask(dim=(4000, 4000))
    #masks = polygon.generateMask2(dim=(4000, 4000), resize=(4000, 4000), saveOnline=True)
