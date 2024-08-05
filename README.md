# cnn_emotion4
cnn_emotion4

# Using library
Since the code uses an old version of keras, it needs to be placed at the beginning of the main.py code.

    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    import BodyEmotion4Lib.Classifier as bec
    from PIL import Image
    
    cls=bec.Emotion4Classifier(model_type='efficientnet_b3');
    
    img_pil = Image.new('RGB', (400,300), 'white');
    
    res=bec.from_img_pil(img_pil);
    
    print(res);

# Installation summary

    git clone https://github.com/trucomanx/cnn_emotion4
    gdown 1fq7tnCK2TONygPVdlTNkbeOiSkRihq1C
    unzip models.zip -d cnn_emotion4/library/BodyEmotion4Lib/models
    cd cnn_emotion4/library
    python3 setup.py sdist
    pip3 install dist/BodyEmotion4Lib-*.tar.gz
