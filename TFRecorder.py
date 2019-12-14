"""
TFRecorder.py
convert numpy ndarray to TFRecorder
Zhiang Chen, Dec 2019
Harish Hanand

Copyright (c) 2019 Distributed Robotic Exploration and Mapping Systems Laboratory, ASU
"""

import tensorflow as tf
from io import BytesIO
from PIL import Image
import os
import io
import PIL
import hashlib
import numpy as np
import matplotlib.pyplot as plt


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature( value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature( value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_examples_list(path):
    """Read list of training or validation examples.

    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.

    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

    Args:
    path: absolute path to examples list file.

    Returns:
    list of example identifiers (strings).
    """
    with tf.io.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]

def recursive_parse_xml_to_dict(xml):
    """
    Recursively parses XML contents to python dict.
    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.
    Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
    Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFConverter(object):
    def __init__(self, object_list, annotation_path, image_path):
        self.object_list = object_list  # 0 is background
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.annotations = [f for f in os.listdir(annotation_path) if f.endswith(".npy")]
        self.images = [i for i in os.listdir(image_path) if i.endswith("jpg")]

    def convert(self, output_path="."):
        writer = tf.io.TFRecordWriter(output_path + "/tf.record")
        for image in self.images:
            annotation = image.split(".")[0] + ".npy"
            if annotation not in self.annotations:
                print("No annotation file. Ignore "+image)
                continue
            else:
                annotation_file = os.path.join(self.annotation_path, annotation)
                image_file = os.path.join(self.image_path, image)
                masks = np.load(annotation_file)
                if masks.max()  < 1:
                    continue
                else:
                    tf_example = self._getTFExample(annotation_file, image_file)
                    writer.write(tf_example.SerializeToString())
        writer.close()



    def _getTFExample(self, annotation_file_path, image_file_path):
        # TODO: modify this function

        fid = tf.io.gfile.GFile(image_file_path, 'rb')  # tensorflow 2.0
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)

        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width = image.width
        height = image.height
        masks = np.load(annotation_file_path)
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, 2)
        obj_nm = masks.shape[-1]
        masks = (masks/255.0).astype(np.uint8)

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        masks_PNG = []
        masks_list = []

        for i in range(obj_nm):
            mask = masks[:, :, i]
            if mask.max() < 1:
                continue
            masks_list.append(mask.tostring())
            bbox = self._getBBox(mask)
            xmin.append(float(bbox[0]) / width)
            ymin.append(float(bbox[2]) / height)
            xmax.append(float(bbox[1]) / width)
            ymax.append(float(bbox[3]) / height)
            """
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            """
            id = int(mask.max())
            classes.append(id)
            classes_text.append(self.object_list[id-1])
            masks_PNG.append(self._image2PNGString(mask))

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(image_file_path.encode('utf8')),  #
            'image/source_id': bytes_feature(image_file_path.encode('utf8')),  #
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/text': bytes_list_feature(classes_text),
            'image/object/class/label': int64_list_feature(classes),
            'image/object/mask': bytes_list_feature(masks_PNG)
            }))
        return example

    def _image2PNGString(self, image):
        image = Image.fromarray(image)
        byte_io = BytesIO()
        image.save(byte_io, 'PNG')
        return byte_io.getvalue()

    def _getBBox(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        xmin, xmax, ymin, ymax = rmin, rmax, cmin, cmax

        return xmin, xmax, ymin, ymax


if __name__ == "__main__":
    tfrecorder = TFConverter(['rock'], "./datasets/rocks", "./datasets/rocks")
    tfrecorder.convert()
