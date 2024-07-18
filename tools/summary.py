#!/usr/bin/python
import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub

DATA=[
    {
    'model_type':'mobilenet_v3',
    'url':'https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5',
    'target_size':(224,224)
    },
    {
    'model_type':'resnet_v2_50',
    'url':'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5',
    'target_size':(224,224)
    },
    {
    'model_type':'efficientnet_b3',
    'url':'https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1',
    'target_size':(300,300)
    },
    {
    'model_type':'inception_v3',
    'url':'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4',
    'target_size':(299,299)
    },
    {
    'model_type':'inception_resnet_v2',
    'url':'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5',
    'target_size':(299,299)
    }
]


for item in DATA:
    m = tf.keras.Sequential([
        hub.KerasLayer(item['url'],
                       trainable=False),  # Can be True, see below.
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    m.build([None, item['target_size'][0], item['target_size'][1], 3])  # Batch input shape.
    
    print('\n\n',item['model_type'])
    m.summary();

