#!/usr/bin/python3

import sys
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable CUDA
os.environ['TF_USE_LEGACY_KERAS']='1' 


## Model of network
#model_type = 'yolov8m-cls'
#model_type = 'yolov8s-cls'
#model_type = 'yolov8n-cls'
model_type = 'efficientnet_b3'
#model_type = 'mobilenet_v3';
#model_type = 'inception_v3';
#model_type = 'inception_resnet_v2';
#model_type = 'resnet_v2_50';

for n in range(len(sys.argv)):
    if sys.argv[n]=='--model':
        model_type=sys.argv[n+1];


print('    model_type:',model_type);

import cv2
import openpifpaf
from PIL import Image
import OpenPifPafTools.OpenPifPafGetData as oppgd

sys.path.append('../library');
from BodyEmotion4Lib.Classifier import Emotion4Classifier

def my_func(Clf,predictor,frame):
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    pil_im = Image.fromarray(rgb_frame);
    
    height, width, channels = frame.shape
    
    annotation, gt_anns, image_meta = predictor.pil_image(pil_im);
    
    for annot in annotation: 
        (xi,yi,xo,yo)=oppgd.get_body_bounding_rectangle(annot.data,factor=1.0);
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
        
        pil_crop=pil_im.crop((xi,yi,xo,yo));
        res=Clf.from_img_pil(pil_crop);
        
        texto=Clf.target_labels()[res];
        
        frame = cv2.putText(  frame,
                              texto,
                              org = (int(xi), int((yi+yo)/2)),
                              fontFace = cv2.FONT_HERSHEY_DUPLEX,
                              fontScale = 2.0,
                              color = (255, 0, 0),
                              thickness = thickness
                            )
        
        cv2.rectangle(frame,(xi,yi),(xo,yo),color,thickness);
    return frame;


Clf=Emotion4Classifier(model_type=model_type,file_of_weight='');

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
#predictor = openpifpaf.Predictor(checkpoint='resnet50')

# Mover o modelo para a CPU
#predictor.model = predictor.model.to('cpu')

# Inicializa la captura de video con la webcam (índice 0)
cap = cv2.VideoCapture(0)


# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Bucle para capturar cuadro por cuadro
while True:
    
    # Captura frame por frame
    ret, frame = cap.read()

    # Si no se lee correctamente el frame, salir del bucle
    if not ret:
        print("Error: No se puede recibir el frame (final de transmisión?)")
        break
    
    # Muestra el frame en una ventana llamada 'Webcam'
    cv2.imshow('Webcam', my_func(Clf,predictor,frame) )

    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Libera la captura de video
cap.release()

# Cierra todas las ventanas
cv2.destroyAllWindows()

