#!/usr/bin/python3
import sys
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable CUDA
os.environ['TF_USE_LEGACY_KERAS']='1' 
## Model of network
model_type = 'efficientnet_b3'
#model_type = 'mobilenet_v3';
#model_type = 'inception_v3';
#model_type = 'inception_resnet_v2';
#model_type = 'resnet_v2_50';


import cv2
import openpifpaf
import OpenPifPafTools.OpenPifPafGetData as oppgd

sys.path.append('../library');
from BodyEmotion4Lib.Classifier import Emotion4Classifier



# Inicializar o OpenPifPaf
predictor = openpifpaf.Predictor(checkpoint='resnet50')
#predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')

Clf=Emotion4Classifier(model_type=model_type,file_of_weight='/media/fernando/Expansion/OUTPUTS/DOCTORADO2/cnn_emotion4/ber2024-body/training_validation_holdout_fine_tuning/efficientnet_b3/model_efficientnet_b3.h5');


# Inicializar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame da webcam
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar imagem")
        break

    # Converter a imagem para RGB (OpenPifPaf espera imagens RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height,width,deep=rgb_frame.shape
    
    # Fazer a predição dos keypoints
    predictions, _, _ = predictor.numpy_image(rgb_frame)

    # Desenhar os keypoints na imagem
    for pred in predictions:
    
        (xi,yi,xo,yo)=oppgd.get_body_bounding_rectangle(pred.data,factor=1.0);
        xi=int(xi);        yi=int(yi);
        xo=int(xo);        yo=int(yo);
        
        if xo<0:
            xo=0;
        if xo>=width:
            xo=width-1;
        if yo<0:
            yo=0;
        if yo>=height:
            yo=height-1;
        
        color=(0,255,0);
        thickness=2;
        
        cv2.rectangle(frame,(xi,yi),(xo,yo),color,thickness);

    # Mostrar a imagem com keypoints
    cv2.imshow('Keypoints', frame)

    # Sair do loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura da webcam e destruir as janelas
cap.release()
cv2.destroyAllWindows()

