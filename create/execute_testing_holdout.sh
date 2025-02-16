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

OutDir='/media/fernando/Expansion/OUTPUTS/DOCTORADO2/cnn_emotion4_1'

SubTitle='fine_tuning'

#DName='perwi'  
DName='ber2024-body'  

if [ "$DName" = "perwi" ]; then
    InTsD='/media/fernando/Expansion/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/test/'
    InTsF='labels-emotion4-v1.csv'
    ModD='/media/fernando/Expansion/OUTPUTS/DOCTORADO2/BODY/cnn_emotion4/perwi/training_validation_holdout_fine_tuning'
fi

if [ "$DName" = "ber2024-body" ]; then
    InTsD='/media/fernando/Expansion/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTsF='test.csv'
    ModD='/media/fernando/Expansion/OUTPUTS/DOCTORADO2/BODY/cnn_emotion4/ber2024-body/training_validation_holdout_fine_tuning'
fi

################################################################################

if [ "$SubTitle" = "" ]; then
    BaseDir='test_holdout'
else
    BaseDir='test_holdout_'$SubTitle
fi

mkdir -p $OutDir/$DName/$BaseDir
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/$BaseDir/'main.py'

################################################################################
export TF_USE_LEGACY_KERAS=1 

ipynb-py-convert testing_holdout.ipynb testing_holdout.py

python3 testing_holdout.py --model 'efficientnet_b3'     --model-file $ModD/'efficientnet_b3/model_efficientnet_b3.h5'         --times 1 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'inception_resnet_v2' --model-file $ModD/'inception_resnet_v2/model_inception_resnet_v2.h5' --times 1 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'inception_v3'        --model-file $ModD/'inception_v3/model_inception_v3.h5'               --times 1 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'mobilenet_v3'        --model-file $ModD/'mobilenet_v3/model_mobilenet_v3.h5'               --times 1 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle
python3 testing_holdout.py --model 'resnet_v2_50'        --model-file $ModD/'resnet_v2_50/model_resnet_v2_50.h5'               --times 1 --batch-size  16 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir --output-subtitle $SubTitle

rm -f testing_holdout.py

