#!/usr/bin/python3

import os

## Because the KERAS used is a old verion, KERAS 1
os.environ['TF_USE_LEGACY_KERAS'] = '1'

## Allows memory allocation on the GPU to be done asynchronously, 
## potentially improving performance and efficiency in certain scenarios.
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

## Specify which GPUs are available to a program.
## The list of GPUs is usually represented by indices starting at 0.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable cuda

import sys
sys.path.append('../library') 

import BodyEmotion4Lib.Classifier as bec
import numpy as np
from PIL import Image

cls=bec.Emotion4Classifier( model_type='efficientnet_b3');


labels   = ['negative','neutral','pain','positive'];

filepaths = [   'KlingAi/negative.png',
                'KlingAi/neutral.png',
                'KlingAi/pain.png',
                'KlingAi/positive.png',
                'KlingAi/pain_negative.png'];

pil_list=[Image.open(filepath) for filepath in filepaths];

res=cls.predict_pil_list(pil_list);
res_arg=np.argmax(res, axis=1);

for n in range(len(filepaths)):
    print(filepaths[n],labels[res_arg[n]],res[n,:]);
    

ID=cls.from_img_pil_list(pil_list);
print(type(ID),ID)

for n in range(len(filepaths)):
    print(filepaths[n],labels[ID[n]],ID[n]);
    

