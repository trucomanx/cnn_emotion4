#!/bin/bash

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

#python3 kfold_validation.py --model 'efficientnet_b3' --epochs 400
#python3 kfold_validation.py --model 'inception_resnet_v2' --epochs 400
#python3 kfold_validation.py --model 'inception_v3' --epochs 400
python3 kfold_validation.py --model 'mobilenet_v3' --epochs 400 --batch-size 128
#python3 kfold_validation.py --model 'resnet_v2_50' --epochs 400

rm -f kfold_validation.py
