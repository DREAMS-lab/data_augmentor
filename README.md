# Data Augmentor
# A python data augmentation tool for deep learning

MIT License
Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

Zhiang Chen, Nov 2018

### 1. xml2mask.py
[xml2mask.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/xml2mask.py) provides methods of generating masks from xml files downloaded from web-based annotation tool [LabelMe](http://labelme.csail.mit.edu). It can
- generate and combine all masks on a single layer and save as `.jpg` files. e.g. mask.shape = [width, height]
- generate all masks on individual layers and save as `.npy` files. e.g. mask.shape = [nm_objects, width, height]

### 2. rename.py
[rename.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/xml2mask.py) provides methods of renaming images and the corresponding annotations by numerical order. It supports to
- rename images and annotations by numerical order
- add annotations with prefix "label_" 
- convert with `.jpg` and `.png`
- work with `.npy`

### 3. augmentation.py
[augmentation.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/augmentation.py) augments images and corresponding annotations with same rules, which include  
- resizing
- left-right flipping
- up-down flipping
- rotating
- zooming-in and zooming-out

### 4. generateCSV.py
[generateCSV.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/generateCSV.py) generates CSV files for RetinaNet https://github.com/DREAMS-lab/keras-retinanet. See the test main for how to use it: https://github.com/DREAMS-lab/data_augmentor/blob/master/generateCSV.py#L67.

### 5. processTiff.py
[processTiff.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/processTiff.py) provides methods of processing GeoTiff. It can
- split large GeoTiff to small GeoTiff
- convert GeoTiff to PNG  

### 6. labelboxReader.py
[labelReader.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/labelboxReader.py) provides methods of generating masks from json files downloaded from web-based annotation tool [LabelBox](https://labelbox.com). 

This is dependent on GDAL python. To install the related packages:
```
$ sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
$ sudo apt-get install gdal-bin
```