#!/bin/bash

mkdir coco
cd coco

# wget -c http://images.cocodataset.org/zips/train2017.zip
# echo "Extracting train2017.zip"
# unzip -qq train2017.zip
# rm train2017.zip

# wget -c http://images.cocodataset.org/zips/val2017.zip
# echo "Extracting val2017.zip"
# unzip -q val2017.zip
# rm val2017.zip

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Extracting annotations_trainval2017.zip"
unzip -q annotations_trainval2017.zip
rm annotations_trainval2017.zip

cd annotations
find . -type f \
    -not -wholename ./instances_train2017.json \
    -not -wholename ./instances_val2017.json \
    -delete

echo "DONE."