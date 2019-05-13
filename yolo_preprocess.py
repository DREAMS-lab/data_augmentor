import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


class yolo_pp(object):
    def __init__(self):
        pass
    
    def generateTXT(self, dataset_path, save_name, form=".jpg"):
        files = [f for f in os.listdir(dataset_path) if f.endswith(form)]
        if len(files)>0:
            dirt = os.path.abspath(dataset_path)
            with open(save_name, 'w') as txt_file:
                for f in files:
                    npy_name = os.path.join(dirt, f).split(".")[0]+".npy"
                    label = np.load(npy_name)
                    if len(label.shape) == 3:
                        w,h,c = label.shape
                        got, coords = self.extractBB(label)
                        if got:
                            name = os.path.join(dirt, f) + '\n'
                            txt_file.write(name)
                txt_file.close()
                
    def generateLabel(self, dataset_path):
        files = [f for f in os.listdir(dataset_path) if f.endswith(".npy")]
        if len(files)>0:
            dirt = os.path.abspath(dataset_path)
            if not os.path.isdir(dataset_path+"../label"):
                os.mkdir(dataset_path+"../label")
            txt_dirt = os.path.abspath(dataset_path+"../label")
            for f in files:
                npy_name = os.path.join(dirt, f)
                txt_name = os.path.join(txt_dirt, f.split('.')[0]+'.txt' )
                label = np.load(npy_name)
                
                if len(label.shape) == 2:  # no object
                    #txt_f = open(txt_name, 'w')
                    #txt_f.close()
                    None
                elif len(label.shape) == 3:
                    h,w,c = label.shape
                    got, coords = self.extractBB(label)
                    if not got:
                        #txt_f.close()
                        None
                    else:
                        txt_f = open(txt_name, 'w')
                        n,c = coords.shape
                        for i in range(c):
                            #x, y = coords[2,i], coords[0,i]
                            #x_, y_ = coords[3,i], coords[1,i]
                            y, y_, x, x_ = coords[:,i]
                            box = (x, x_, y, y_)
                            bb = convert((w,h), box)
                            txt = "0"+ " " + " ".join([str(a) for a in bb]) + '\n'
                            txt_f.write(txt)
                        txt_f.close()
                        
    def displayBB(self, masks, coords):
        mask = masks.max(2)
        fig,ax = plt.subplots(1)
        # Display the image
        ax.imshow(mask)
        n,c = coords.shape
        for i in range(c):
            # Create a Rectangle patch
            x, y = coords[2,i], coords[0,i]
            w, h = coords[3,i] - coords[2,i], coords[1,i] - coords[0,i]
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()
        
                        
                
    def extractBB(self, mask):
        if mask.max() < 1:
            return False, None
        else:
            y, x, c = mask.shape
            coords = np.zeros((4,c))
            for i in range(c):
                img = mask[:,:,i]
                coord = self.__bbox(img)
                coords[:,i] = coord
            return True, coords
        
        
    def __bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return np.array((rmin, rmax, cmin, cmax))  # ymin, ymax, xmin, xmax
        
    
if __name__  ==  "__main__":
    yolopp = yolo_pp()
    yolopp.generateTXT("./datasets/drone/train/", "./datasets/drone/drone_train.txt")
    yolopp.generateTXT("./datasets/drone/valid/", "./datasets/drone/drone_valid.txt")
    yolopp.generateLabel("./datasets/drone/train/")
    yolopp.generateLabel("./datasets/drone/valid/")
