"""
multispectrum
Zhiang Chen,
Feb, 2020
"""

import gdal
import cv2
import numpy as np
import math
import os

class MultDim(object):
    def __init__(self):
        pass

    def readTiff(self, tif_file, channel=3):
        self.ds = gdal.Open(tif_file)
        B = self.ds.GetRasterBand(1).ReadAsArray()
        G = self.ds.GetRasterBand(2).ReadAsArray()
        R = self.ds.GetRasterBand(3).ReadAsArray()
        if channel ==3:
            cv2.imwrite("./datasets/Rock/R.png", R)
            cv2.imwrite("./datasets/Rock/G.png", G)
            cv2.imwrite("./datasets/Rock/B.png", B)

        if channel == 5:
            RE = self.ds.GetRasterBand(4).ReadAsArray()
            NIR = self.ds.GetRasterBand(5).ReadAsArray()
            cv2.imwrite("./datasets/Rock/R.png", R)
            cv2.imwrite("./datasets/Rock/G.png", G)
            cv2.imwrite("./datasets/Rock/B.png", B)
            cv2.imwrite("./datasets/Rock/RE.png", RE)
            cv2.imwrite("./datasets/Rock/NIR.png",NIR)


    def readImage(self, image_file, channel=3):
        if channel==1:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            img = np.expand_dims(img, axis=2)
        else:
            img = cv2.imread(image_file).astype(np.uint8)
        return img

    def cat(self, data1, data2):
        return np.append(data1, data2, axis=2)

    def split(self, data, step, path, overlap=0):
        dim = data.shape
        mult = np.zeros((dim[0]+step, dim[1]+step, dim[2]))
        mult[:dim[0], :dim[1], :] = data
        xn = int(math.ceil(float(dim[0])/(step-overlap)))
        yn = int(math.ceil(float(dim[1])/(step-overlap)))
        for i in range(xn):
            for j in range(yn):
                x = i*(step-overlap)
                y = j*(step-overlap)
                dt = mult[x:x+step, y:y+step, :]
                name = os.path.join(path, str(i)+"_"+str(j)+".npy")
                np.save(name, dt)

    def addAnnotation(self, mult_path, annotation_path, save_path):
        ann_files = os.listdir(annotation_path)
        mult_files = os.listdir(mult_path)
        for f in ann_files:
            if f in mult_files:
                ann_name = os.path.join(annotation_path, f)
                mult_name = os.path.join(mult_path, f)
                ann = np.load(ann_name)
                mult = np.load(mult_name)
                data = np.append(mult, ann, axis=2)
                save_name = os.path.join(save_path, f)
                np.save(save_name, data)




if __name__ == '__main__':
    st = MultDim()
    # split tiles
    """
    st.readTiff("./datasets/C3/Orth5.tif", channel=5)
    R = st.readImage("./datasets/Rock/R.png", channel=1)
    G = st.readImage("./datasets/Rock/G.png", channel=1)
    B = st.readImage("./datasets/Rock/B.png", channel=1)
    RE = st.readImage("./datasets/Rock/RE.png", channel=1)
    NIR = st.readImage("./datasets/Rock/NIR.png", channel=1)
    DEM = st.readImage("./datasets/Rock/DEM3.png", channel=3)
    data = st.cat(R, G)
    data = st.cat(data, B)
    data = st.cat(data, RE)
    data = st.cat(data, NIR)
    data = st.cat(data, DEM)
    st.split(data, 400, "./datasets/Rock/mult_10", overlap=10)
    """
    # add annotations
    # st.addAnnotation("./datasets/Rock/mult/", "./datasets/Rock_test/npy/", "./datasets/Rock_test/mult")
    #"""
    RGB = st.readImage("./datasets/C3/C3.png", channel=3)
    DEM = st.readImage("./datasets/C3/C3_dem.png", channel=3)
    data = st.cat(RGB, DEM)
    st.split(data, 400, './datasets/C3/rgbd', overlap=10)
    #"""
    #st.addAnnotation("./datasets/C3/rgbd/", "./datasets/C3_test/npy/", "./datasets/C3_test/rocks")