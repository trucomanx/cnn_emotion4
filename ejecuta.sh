#!/bin/bash

python3 kfold_validation.py --model 'efficientnet_b3'
python3 kfold_validation.py --model 'mobilenet_v3'
python3 kfold_validation.py --model 'inception_v3'
python3 kfold_validation.py --model 'inception_resnet_v2'
python3 kfold_validation.py --model 'resnet_v2_50'
