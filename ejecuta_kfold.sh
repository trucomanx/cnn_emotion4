#!/bin/bash

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

#python3 kfold_validation.py --model 'efficientnet_b3'
python3 kfold_validation.py --model 'inception_resnet_v2'
python3 kfold_validation.py --model 'inception_v3'
python3 kfold_validation.py --model 'mobilenet_v3'
python3 kfold_validation.py --model 'resnet_v2_50'

rm -f kfold_validation.py
