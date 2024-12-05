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

erro_bar=[("mean_val_categorical_accuracy","std_val_categorical_accuracy")];

sort_by="val_categorical_accuracy";

p_matrix="val_categorical_accuracy";

sep=",";

image_ext=".eps";
'

OutDir='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4_v2'

#DName='perwi' 
DName='ber2024-body'


if [ "$DName" = "ber2024-body" ]; then
    InTrD='/mnt/8811f502-ae19-4dd8-8371-f1915178f581/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTrF='train_refface.csv'
fi

################################################################################

mkdir -p $OutDir/$DName/cross-validation_full-learning
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/cross-validation_full-learning/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert kfold_validation.ipynb kfold_validation.py

python3 kfold_validation.py --model 'efficientnet_b3'     --epochs 100 --batch-size 16 --full-learning true --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'inception_resnet_v2' --epochs 100 --batch-size 16 --full-learning true --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'inception_v3'        --epochs 100 --batch-size 16 --full-learning true --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'mobilenet_v3'        --epochs 100 --batch-size 16 --full-learning true --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir
python3 kfold_validation.py --model 'resnet_v2_50'        --epochs 100 --batch-size 16 --full-learning true --dataset-dir $InTrD --dataset-file $InTrF --dataset-name $DName --output-dir $OutDir


rm -f kfold_validation.py

