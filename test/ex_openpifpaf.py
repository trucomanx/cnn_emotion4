#!/usr/bin/python3

import cv2
import openpifpaf

# Inicializar o OpenPifPaf
#predictor = openpifpaf.Predictor(checkpoint='resnet50')
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')


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

    # Fazer a predição dos keypoints
    predictions, _, _ = predictor.numpy_image(rgb_frame)

    # Desenhar os keypoints na imagem
    for pred in predictions:
        keypoints = pred.data[:, :2]
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Mostrar a imagem com keypoints
    cv2.imshow('Keypoints', frame)

    # Sair do loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura da webcam e destruir as janelas
cap.release()
cv2.destroyAllWindows()

