#!/usr/bin/python
import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub
import pprint


DATA=[
    {
    'model_type':'efficientnet_b3',
    'url':'https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1',
    'target_size':(300,300)
    },
    {
    'model_type':'inception_resnet_v2',
    'url':'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5',
    'target_size':(299,299)
    },
    {
    'model_type':'inception_v3',
    'url':'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4',
    'target_size':(299,299)
    },
    {
    'model_type':'mobilenet_v3',
    'url':'https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5',
    'target_size':(224,224)
    },
    {
    'model_type':'resnet_v2_50',
    'url':'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5',
    'target_size':(224,224)
    }
]


INFO=[];

for item in DATA:
    model = tf.keras.Sequential([
        hub.KerasLayer(item['url'],trainable=False),  # Can be True, see below.
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.build([None, item['target_size'][0], item['target_size'][1], 3])  # Batch input shape.
    
    print('\n\n')
    print(item['model_type'])
    
    model.summary();

    info=dict();
    
    info['model_type'] = item['model_type'];
    info['target_size'] = item['target_size'];
    info['feature_extrator_params'] = model.layers[0].count_params();
    info['total_params'] = model.count_params();
    
    INFO.append(info);
    
    pprint.pprint(info)

pprint.pprint(INFO)
################################################################################


import pandas as pd

# Criando um DataFrame a partir dos dados
df = pd.DataFrame(INFO)

# Selecionando as colunas 'model_type' e 'total_params'
df = df[['model_type', 'total_params','target_size']]

df = df.rename(columns={'model_type': 'Arquitetura', 'total_params': 'Número de parameters', 'target_size': 'Tamanho da entrada'})

# Substituindo todas as ocorrências de 'mobilenet_v3' por 'MobileNetV3'
df = df.replace('efficientnet_b3', 'EfficientNet-B3')
df = df.replace('inception_v3', 'InceptionV3')
df = df.replace('inception_resnet_v2', 'Inception-ResNet-V2')
df = df.replace('mobilenet_v3', 'MobileNetV3')
df = df.replace('resnet_v2_50', 'ResNet-V2-50')

# Convertendo o DataFrame para formato LaTeX
latex_table = df.to_latex(index=False, caption='Número total de parâmetros em cada arquitetura.', label='tab:model_params', column_format='lrr')

# Salvando a tabela LaTeX em um arquivo
with open('output_table.tex', 'w') as f:
    f.write(latex_table)

print(latex_table) 

