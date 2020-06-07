# python3
# This is for annotation convert.
# It can convert VOC xml format to YOLO txt format.
# YOLO txt format can be trained in YOLOv4(darknet exe file).
# 
# The experiment dir is VOC2007.

import sys
import os

from absl import app, flags
from absl.flags import FLAGS
from lxml import etree

flags.DEFINE_string(
    'anno_dir', '/Users/yq/Downloads/datasets/trainval/VOC2007/Annotations', 'path to anno dir')
flags.DEFINE_string(
    'image_dir', '/Users/yq/Downloads/datasets/datasets/trainval/VOC2007/JPEGImages', 'path to image dir')
flags.DEFINE_string(
    'trainval_list_txt', '/Users/yq/Downloads/datasets/trainval/VOC2007/ImageSets/Main/trainval.txt', 'path to a set of train')
flags.DEFINE_string('classes', '/Users/yq/Downloads/datasets/trainval/train_classes.txt',
                    'path to a list of class names')
flags.DEFINE_string('txt_anno_dir', '/Users/yq/Downloads/datasets/trainval/VOC2007/Annotations-txt',
                    'path to a list of class names')

def convert_annotation(list_txt, output_path, image_dir, anno_dir, class_names, anno_dir_txt):
    # print(class_names)
    IMAGE_EXT = '.jpg'
    ANNO_EXT = '.xml'

    with open(list_txt, 'r') as f:
        while True:
            
            line = f.readline().strip()
            # print("processing:", line)
            if line is None or not line:
                break
            im_p = os.path.join(image_dir, line + IMAGE_EXT)
            an_p = os.path.join(anno_dir, line + ANNO_EXT)

            # Get annotation.
            root = etree.parse(an_p).getroot()
            bboxes = root.xpath('//object/bndbox')
            names = root.xpath('//object/name')
            
            widthElms = root.xpath('//size/width')
            heigthElms = root.xpath('//size/height')
            
            imgWidth = int(widthElms[0].text)
            imgHeight = int(heigthElms[0].text)
            # print(imgWidth, imgHeight)
            # print('bboxes:', bboxes)
            # print('names:', names)


            box_annotations = []
            for b, n in zip(bboxes, names):
                name = n.text
                class_idx = class_names.index(name)

                xmin = float(b.find('xmin').text)
                ymin = float(b.find('ymin').text)
                xmax = float(b.find('xmax').text)
                ymax = float(b.find('ymax').text)
                # print("xmin, xmax, ymin, ymax:", xmin, xmax, ymin, ymax)
                # convert to YOLO format
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                # print("center_x, center_y", center_x, center_y)
                width = xmax - xmin
                height = ymax - ymin
                
            
                norm_center_x = center_x / imgWidth
                norm_center_y = center_y / imgHeight
                
                # print("norm_center_x, norm_center_y", norm_center_x, norm_center_y)
                norm_width = width / imgWidth
                norm_height = height / imgHeight
                # print(str(class_idx), str(norm_center_x), str(
                #     norm_center_y), str(norm_width), str(norm_height))
                box_annotations.append(
                    ' '.join([str(class_idx), str(norm_center_x), str(norm_center_y), str(norm_width), str(norm_height)])+'\n')

            # print(box_annotations)
            # write into txt annotation file
            filename = anno_dir_txt + '/' + line + ".txt"
            # print(filename)
            anno_f = open(filename, 'w')
            anno_f.writelines(box_annotations)
            anno_f.close()

def main(_argv):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    convert_annotation(FLAGS.trainval_list_txt, FLAGS.anno_dir,
                       FLAGS.image_dir, FLAGS.anno_dir, class_names, FLAGS.txt_anno_dir)
    print("Convert end.")

if __name__ == "__main__":
    try:
        app.run(main)
    except OSError as err:
        print(err)
