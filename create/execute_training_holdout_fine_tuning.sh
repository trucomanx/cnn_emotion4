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
            "number_of_parameters"
            ];

sep=",";

image_ext=".eps";
'

OutDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4_2'

#DName='perwi'  
DName='ber2024-body'  


if [ "$DName" = "perwi" ]; then
    InTrD='/media/fernando/Expansion/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/train/'
    InTrF='labels-emotion4-v1.csv'
    InTsD='/media/fernando/Expansion/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/test/'
    InTsF='labels-emotion4-v1.csv'
fi

if [ "$DName" = "ber2024-body" ]; then
    InTrD='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTrF='train.csv'
    InTsD='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTsF='test.csv'
fi

################################################################################

mkdir -p $OutDir/$DName/training_validation_holdout_fine_tuning
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/training_validation_holdout_fine_tuning/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert training_holdout.ipynb training_holdout.py

python3 training_holdout.py --model 'efficientnet_b3'     --seed 3 --fine-tuning true --epochs 100 --batch-size  16 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'inception_resnet_v2' --fine-tuning true --epochs 100 --batch-size  32 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'inception_v3'        --fine-tuning true --epochs 100 --batch-size  32 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'mobilenet_v3'        --fine-tuning true --epochs 100 --batch-size  32 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir
#python3 training_holdout.py --model 'resnet_v2_50'        --fine-tuning true --epochs 100 --batch-size  32 --dataset-train-dir $InTrD --dataset-train-file $InTrF --dataset-test-dir $InTsD --dataset-test-file $InTsF --dataset-name $DName --output-dir $OutDir

rm -f training_holdout.py

