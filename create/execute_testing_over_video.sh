#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="testing_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50"
            ];

info_list=[ "block_delayms",
            "categorical_accuracy",
            "loss"];

sep=",";

image_ext=".eps";
'

InFile='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-VIDEOS/dataset-toy/drhouse_mini_cut.mp4'

OutDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4_1'

#DName='perwi'  
DName='ber2024-body'  

WDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4'


################################################################################

mkdir -p $OutDir/$DName/test_over_video
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/test_over_video/'main.py'

################################################################################

ipynb-py-convert testing_over_video.ipynb testing_over_video.py

python3 testing_over_video.py --model 'efficientnet_b3'     --weights-file $WDir/$DName/'training_validation_holdout'/'efficientnet_b3'/'model_efficientnet_b3.h5'         --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'inception_resnet_v2' --weights-file $WDir/$DName/'training_validation_holdout'/'inception_resnet_v2'/'model_inception_resnet_v2.h5' --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'inception_v3'        --weights-file $WDir/$DName/'training_validation_holdout'/'inception_v3'/'model_inception_v3.h5'               --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'mobilenet_v3'        --weights-file $WDir/$DName/'training_validation_holdout'/'mobilenet_v3'/'model_mobilenet_v3.h5'               --dataset-name $DName --output-dir $OutDir --input-file $InFile
python3 testing_over_video.py --model 'resnet_v2_50'        --weights-file $WDir/$DName/'training_validation_holdout'/'resnet_v2_50'/'model_resnet_v2_50.h5'               --dataset-name $DName --output-dir $OutDir --input-file $InFile

rm -f testing_over_video.py

