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

OutDir='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4_1'

DName='perwi'  
#DName='ber2024-body'  

if [ "$DName" = "perwi" ]; then
    InTsD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/PATIENT-RECOGNITION/PATIENT-IMAGES/perwi/dataset/test/'
    InTsF='labels-emotion4-v1.csv'
    ModD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4/perwi/training_validation_holdout'
fi

if [ "$DName" = "ber2024-body" ]; then
    InTsD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/DATASET/TESE/BER/BER2024/BER2024-BODY'
    InTsF='test.csv'
    ModD='/media/fernando/B0EA304AEA300EDA/Dados/Fernando/OUTPUTS/DOCTORADO2/cnn_emotion4/ber2024-body/training_validation_holdout'
fi

################################################################################

mkdir -p $OutDir/$DName/test_holdout
echo "$PyCommand" | cat - 'main.py' > temp && mv temp $OutDir/$DName/test_holdout/'main.py'

################################################################################

ipynb-py-convert testing_holdout.ipynb testing_holdout.py

python3 testing_holdout.py --model 'efficientnet_b3'     --model-dir $ModD/'efficientnet_b3'     --times 10 --batch-size  32 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 testing_holdout.py --model 'inception_resnet_v2' --model-dir $ModD/'inception_resnet_v2' --times 10 --batch-size  64 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 testing_holdout.py --model 'inception_v3'        --model-dir $ModD/'inception_v3'        --times 10 --batch-size  64 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 testing_holdout.py --model 'mobilenet_v3'        --model-dir $ModD/'mobilenet_v3'        --times 10 --batch-size  64 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir
python3 testing_holdout.py --model 'resnet_v2_50'        --model-dir $ModD/'resnet_v2_50'        --times 10 --batch-size  64 --dataset-dir $InTsD --dataset-file $InTsF --dataset-name $DName --output-dir $OutDir

rm -f testing_holdout.py

