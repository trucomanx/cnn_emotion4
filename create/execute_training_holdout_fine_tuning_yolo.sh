#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50",
            "yolov8n-cls",
            "yolov8s-cls",
            "yolov8m-cls"
            ];

info_list=[ "train_categorical_accuracy",
            "val_categorical_accuracy",
            "test_categorical_accuracy"
            ];

sep=",";

image_ext=".eps";
'

OutDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4_1'


DName='ber2024-body'  

if [ "$DName" = "ber2024-body" ]; then
    DatasetD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/YOLO-COPY/TESE/BER/BER2024/BER2024-BODY/'
fi

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout_fine_tuning
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout_fine_tuning/'main.py'

################################################################################

ipynb-py-convert training_holdout_yolo.ipynb training_holdout_yolo.py

python3 training_holdout_yolo.py --model 'yolov8n-cls' --epochs 300 --batch-size  8 --dataset-dir $DatasetD --dataset-name $DName --output-dir $OutDir
python3 training_holdout_yolo.py --model 'yolov8s-cls' --epochs 300 --batch-size  8 --dataset-dir $DatasetD --dataset-name $DName --output-dir $OutDir
python3 training_holdout_yolo.py --model 'yolov8m-cls' --epochs 300 --batch-size  8 --dataset-dir $DatasetD --dataset-name $DName --output-dir $OutDir

rm -f training_holdout_yolo.py

