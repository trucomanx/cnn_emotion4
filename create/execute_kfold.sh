#!/bin/bash

PyCommand='#!/usr/bin/python3

json_filename="kfold_data_results.json"

model_list=["efficientnet_b3",
            "inception_resnet_v2",
            "inception_v3",
            "mobilenet_v3",
            "resnet_v2_50"
            ];

info_list=[ "mean_val_categorical_accuracy",
            "std_val_categorical_accuracy",
            "mean_val_loss",
            "mean_train_categorical_accuracy",
            "mean_train_loss"];

sep=",";

image_ext=".eps";
'

OutDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4_1'

#DName='perwi' 
DName='ber2024-body'


if [ "$DName" = "perwi" ]; then
    InTrD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/train'
    InTrF='labels-emotion4-v1.csv'
fi

if [ "$DName" = "ber2024-body" ]; then
    InTrD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTrF='train.csv'
fi

################################################################################

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

#python3 kfold_validation.py --model 'efficientnet_b3'     --epochs 75 --batch-size  32 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
#python3 kfold_validation.py --model 'inception_resnet_v2' --epochs 75 --batch-size  64 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'inception_v3'        --epochs 75 --batch-size  64 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'mobilenet_v3'        --epochs 75 --batch-size 128 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'resnet_v2_50'        --epochs 75 --batch-size  64 --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir

cp 'main.py' $OutDir/'main.py'
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation/'main.py'

rm -f kfold_validation.py

