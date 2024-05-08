#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="training_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50"
            ];

info_list=[ "train_categorical_accuracy",
            "val_categorical_accuracy",
            "test_categorical_accuracy",
            "train_loss",
            "val_loss",
            "test_loss"];

sep=",";

image_ext=".eps";
'

OutDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4_1'

#DName='perwi'  
DName='ber2024-body'  


if [ "$DName" = "perwi" ]; then
    InTrD='/media/fernando/Expansion/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/train/'
    InTrF='labels-emotion4-v1.csv'
    InTsD='/media/fernando/Expansion/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/test/'
    InTsF='labels-emotion4-v1.csv'
fi

if [ "$DName" = "ber2024-body" ]; then
    InTrD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTrF='train.csv'
    InTsD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTsF='test.csv'
fi

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout/'main.py'

################################################################################

ipynb-py-convert training_holdout.ipynb training_holdout.py

#python3 training_holdout.py --model 'efficientnet_b3'     --epochs 100 --batch-size  32 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'inception_resnet_v2' --epochs 100 --batch-size  64 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'inception_v3'        --epochs 100 --batch-size  64 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 training_holdout.py --model 'mobilenet_v3'        --epochs 100 --batch-size  64 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 training_holdout.py --model 'resnet_v2_50'        --epochs 100 --batch-size  64 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir

rm -f training_holdout.py

