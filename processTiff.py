"""
processTiff.py
Zhiang Chen, Feb 2019

Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
GeoTiff splitting is done by gdal_translate https://www.gdal.org/gdal_translate.html
"""

import os
import gdal
import cv2
import pickle
import numpy as np
from osgeo import gdal
from osgeo import osr

class splitTiff(object):
    def __init__(self):
        pass

    def readTiff(self, tif):
        ds = gdal.Open(tif)
        band = ds.GetRasterBand(1)
        self.tif = tif
        self.xsize = band.XSize
        self.ysize = band.YSize
        os.system("gdalinfo " + tif)

    def split(self, tile_x, tile_y, overlap_x, overlap_y, save_folder):
        for i in range(0, self.xsize, tile_x - overlap_x):
            for j in range(0, self.ysize, tile_y - overlap_y):
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(
                    tile_x) + ", " + str(tile_y) + " " + str(self.tif) + " " + str(
                    save_folder) + str(i) + "_" + str(j) + ".tif"
                os.system(com_string)


    def convert2png(self, tif_folder, save_folder, rename = False):
        tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif")]
        file_dict = dict()
        for i, tif in enumerate(tif_files):
            f = tif_folder + tif
            img = cv2.imread(f)
            if rename is False:
                out_f = save_folder + tif.split('.')[0] + '.png'
                cv2.imwrite(out_f, img)
            if rename is True:
                out_f = save_folder + str(i) + '.png'
                cv2.imwrite(out_f, img)
                file_dict.setdefault(str(i) + '.png', tif)
        if rename is True:
            with open(save_folder + "png2tiff_dict", 'wb') as f:
                pickle.dump(file_dict, f)

    def convert2jpg(self, tif_folder, save_folder, rename = False):
        tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif")]
        file_dict = dict()
        for i, tif in enumerate(tif_files):
            f = tif_folder + tif
            img = cv2.imread(f)
            if rename is False:
                out_f = save_folder + tif.split('.')[0] + '.jpg'
                cv2.imwrite(out_f, img)
            if rename is True:
                out_f = save_folder + str(i) + '.jpg'
                cv2.imwrite(out_f, img)
                file_dict.setdefault(str(i) + '.jpg', tif)
        if rename is True:
            with open(save_folder + "jpg2tiff_dict", 'wb') as f:
                pickle.dump(file_dict, f)

    def lookupTiff(self, pickle_f, png):
        with open(pickle_f, 'rb') as f:
            file_dict = pickle.load(f)
        print(file_dict[png])


class catTiff(object):
    def __init__(self):
        pass

    def readImages(self, folder, prefix):
        self.folder = folder
        self.prefix = prefix

        self.images = [self.folder+x for x in os.listdir(self.folder) if x.startswith(self.prefix)]
        self.names = [x.split(self.prefix)[-1] for x in os.listdir(self.folder) if x.startswith(self.prefix)]

    def cat(self, overlap_x, overlap_y, resize=None, detect_boundary=False):
        dim = cv2.imread(self.images[0]).shape

        y, x = dim[:2]
        if resize != 0:
            y, x = resize

        suffix = self.images[0].split('.')[-1]

        ul = list()
        for name in self.names:
            name = name.split('.')[0].split('_')
            upper_left = [int(name[0]), int(name[1])]
            ul.append(upper_left)
        ul = np.asarray(ul)
        min_x = np.min(ul[:,0])
        min_y = np.min(ul[:,1])
        max_x = np.max(ul[:,0])
        max_y = np.max(ul[:,1])
        X = max_x-min_x+x
        Y = max_y-min_y+y
        IMAGE = np.zeros((Y, X, 3))
        print(IMAGE.shape)
        for i in range(min_x, max_x, x-overlap_x):
            for j in range(min_y, max_y, y-overlap_y):
                name = self.folder + self.prefix + str(i) + '_' + str(j) + '.' + suffix
                image = cv2.imread(name)
                if resize != None:
                    image = cv2.resize(image,dsize=resize)
                if not detect_boundary:
                    IMAGE[j:j+y, i:i+x, :] = image

        return IMAGE

    def writeTiff(self, data, folder, name, metadata_source, copy_meta=True):
        if copy_meta:
            y,x,c = data.shape
            ds = gdal.Open(metadata_source)
            md = ds.GetMetadata()
            gt = ds.GetGeoTransform()
            pj = ds.GetProjection()


            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(
                folder+name,
                x,
                y,
                c,
                gdal.GDT_Byte)

            dataset.SetGeoTransform(gt)
            dataset.SetMetadata(md)
            dataset.SetProjection(pj)
            dataset.GetRasterBand(1).WriteArray(data[:,:,0])
            dataset.GetRasterBand(2).WriteArray(data[:,:,1])
            dataset.GetRasterBand(3).WriteArray(data[:,:,2])
            dataset.FlushCache()





if __name__  ==  "__main__":
    st = splitTiff()
    #st.readTiff("./datasets/C3/C3.tif")
    #st.split(400, 400, 10, 10, "./datasets/C3/split/")
    st.convert2jpg("./datasets/C3/split/", "./datasets/C3/valid/")
    #st.lookupTiff("./datasets/C3/valid/png2tiff_dict", '3.png')
"""
from processTiff import catTiff
ct = catTiff()
ct.readImages("./datasets/C3/masks/","masked_")
Image = ct.cat(10,10,resize=(400,400))
ct.writeTiff(Image, "./datasets/C3/", "C3_mask.tif", "./datasets/C3/C3.tif")
"""