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



if __name__  ==  "__main__":
    st = splitTiff()
    #st.readTiff("./datasets/C3/C3.tif")
    #st.split(400, 400, 10, 10, "./datasets/C3/split/")
    st.convert2jpg("./datasets/C3/split/", "./datasets/C3/valid/")
    #st.lookupTiff("./datasets/C3/valid/png2tiff_dict", '3.png')