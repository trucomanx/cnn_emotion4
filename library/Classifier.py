#!/usr/bin/python

import os
import lib_model as mpp
import PIL

class Emotion4Classifier:
    def __init__(self,model_type='efficientnet_b3'):
        checkpoint_path='';
        self.model_type=model_type;
        
        if self.model_type=='mobilenet_v3':
            checkpoint_path=os.path.join('models','model_mobilenet_v3.h5');
        elif self.model_type=='resnet_v2_50':
            checkpoint_path=os.path.join('models','model_resnet_v2_50.h5');
        elif self.model_type=='efficientnet_b3':
            checkpoint_path=os.path.join('models','model_efficientnet_b3.h5');
        elif self.model_type=='inception_v3':
            checkpoint_path=os.path.join('models','model_inception_v3.h5');
        elif self.model_type=='inception_resnet_v2':
            checkpoint_path=os.path.join('models','model_inception_resnet_v2.h5');
        else:
            raise TypeError("Unknown parameter model_type");
        
        path = os.path.dirname(__file__)
        path = os.path.join(path,checkpoint_path);
        
        self.modelo, self.target_size=mpp.create_model(path,model_type=self.model_type);
    
    def is_file_patient(self,imgfilepath):
        return mpp.evaluate_model_from_file(self.modelo,imgfilepath, target_size=self.target_size);

    def is_pil_patient(self,img_pil):
        return mpp.evaluate_model_from_pil(self.modelo,img_pil, target_size=self.target_size);
    
    def target_labels(self):
        return ['Negative','Neutro','Pain','Positive'];
