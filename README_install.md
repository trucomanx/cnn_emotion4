# Packaging


Download the source code

    
    git clone https://github.com/trucomanx/cnn_emotion4


Download the models from cnn_emotion4/ber2024-body/fine_tuning
    
    gdown 1fq7tnCK2TONygPVdlTNkbeOiSkRihq1C
    unzip models.zip -d cnn_emotion4/library/BodyEmotion4Lib/models
    

The next command generates the `dist/BodyEmotion4Lib-VERSION.tar.gz` file.

    cd cnn_emotion4/library
    python3 setup.py sdist

For more informations use `python setup.py --help-commands`

# Install 

Install the packaged library

    pip3 install dist/BodyEmotion4Lib-*.tar.gz

# Uninstall

    pip3 uninstall BodyEmotion4Lib
