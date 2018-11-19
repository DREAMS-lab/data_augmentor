# Data Augmentor
# A python data augmentation tool for deep learning

MIT License
Copyright (c) 2018 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU

Zhiang Chen, Nov 2018

### 1. xml2mask.py
[xml2mask.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/xml2mask.py) provides methods of generating masks from xml files that downloaded from web-based annotation tool [LabelMe](http://labelme.csail.mit.edu). It can
- generate and combine all masks on a single layer and save as `.jpg` files. e.g. mask.shape = [width, height]
- generate all masks on individual layers and save as `.npy` files. e.g. mask.shape = [nm_objects, width, height]

### 2. rename.py
[rename.py](https://github.com/DREAMS-lab/data_augmentor/blob/master/xml2mask.py) provides methods of renaming images and the corresponding annotations by numerical order. It supports to
- rename images and annotations by numerical order
- add annotations with prefix "label_" 
- convert with `.jpg` and `.png`
- work with `.npy`
