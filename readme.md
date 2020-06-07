# YOLO Format
Converting other annotation formats to YOLO format. Annotation is used in object detection projects.

## To convert Open Image Dataset format to YOLO format
See OID directory.

## To convert VOC(xml) format to YOLO format
See [annotation_convert_voc_to_yolo.py](./VOC2007/annotation_convert_voc_to_yolo.py)

this is the yolo format
```bash
# center-x center-y width height are all normalized.
# className center-x center-y width height
0 0.479375 0.49130434782608695 0.85375 0.9721739130434782
```

## Checking the converted annotation via LabelIMG
Putting the images and annotations into one folder together, and add a classes.txt file which contains the classes names. Then open the folder with LabelIMG tool, to see if the bounding box are correct.

## Convert VOC(xml) to one list file with file path and annotations
see [voc_convert.py](./voc_convert.py)
the result is like this:

```
/Users/yq/Downloads/datasets/trainval/VOC2007/JPEGImages/20190816_095426.jpg 447,328,3443,2757,1
/Users/yq/Downloads/datasets/trainval/VOC2007/JPEGImages/20190816_095457.jpg 502,1,4032,2905,1
/Users/yq/Downloads/datasets/trainval/VOC2007/JPEGImages/20190816_095522.jpg 95,131,3747,3024,1
```