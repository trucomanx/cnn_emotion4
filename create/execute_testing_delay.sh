#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="testing_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50",
            "yolov8n-cls",
            "yolov8s-cls",
            "yolov8m-cls"
            ];

info_list=[ "delayms",
            "categorical_accuracy"];

sep=",";

image_ext=".eps";
'

OutDir='/home/fernando/Downloads/cnn_emotion4_1'

#DName='perwi'  
DName='ber2024-body'  

if [ "$DName" = "perwi" ]; then
    InTsD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/test/'
    InTsF='labels-emotion4-v1.csv'
    ModD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4/perwi/training_validation_holdout'
fi

if [ "$DName" = "ber2024-body" ]; then
    InTsD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTsF='test.csv'
    ModD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4/ber2024-body/training_validation_holdout_fine_tuning'
fi

################################################################################

mkdir -p $OutDir/$DName/test_delay
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/test_delay/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert testing_delay.ipynb testing_delay.py

python3 testing_delay.py --model 'efficientnet_b3'     --model-dir $ModD/'efficientnet_b3'     --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo False
python3 testing_delay.py --model 'inception_resnet_v2' --model-dir $ModD/'inception_resnet_v2' --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo False
python3 testing_delay.py --model 'inception_v3'        --model-dir $ModD/'inception_v3'        --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo False
python3 testing_delay.py --model 'mobilenet_v3'        --model-dir $ModD/'mobilenet_v3'        --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo False
python3 testing_delay.py --model 'resnet_v2_50'        --model-dir $ModD/'resnet_v2_50'        --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo False

python3 testing_delay.py --model 'yolov8n-cls'        --model-dir $ModD/'yolov8n-cls'          --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo True
python3 testing_delay.py --model 'yolov8s-cls'        --model-dir $ModD/'yolov8s-cls'          --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo True
python3 testing_delay.py --model 'yolov8m-cls'        --model-dir $ModD/'yolov8m-cls'          --times 5 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --yolo True

rm -f testing_delay.py

