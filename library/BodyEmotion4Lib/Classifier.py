#!/usr/bin/python

import os
import BodyEmotion4Lib.lib_model as mpp

import PIL

class Emotion4Classifier:
    """Class to classify 4 body languages.
    
    The class Emotion4Classifier classify daa in 4 body languages.
    
    Args:
        model_type: Type of model architecture. 
        This can be: 
        'efficientnet_b3', 'inception_resnet_v2', 'inception_v3', 'mobilenet_v3', 'resnet_v2_50', 
        'yolov8n-cls'
        
    Atributos:
        modelo: Model returned by tensorflow.
        model_type: Architecture of model.
        target_size: Input size of model. The input image will be reshape to (target_size,target_size,3).
    """
    def __init__(self,model_type='efficientnet_b3',file_of_weight=''):
        """Inicializer of class Emotion4Classifier.
        
        Args:
            model_type: Type of model architecture. 
            This can be: 
            'efficientnet_b3', 'inception_resnet_v2', 'inception_v3', 'mobilenet_v3', 'resnet_v2_50', 
            'yolov8n-cls'
        """
        self.model_type=model_type;
        
        self.model_list_hub  = ['efficientnet_b3', 'inception_resnet_v2', 'inception_v3', 'mobilenet_v3', 'resnet_v2_50'];
        self.model_list_yolo = ['yolov8n-cls','yolov8s-cls','yolov8m-cls'];
        
        if len(file_of_weight)>0:
            if   self.model_type in self.model_list_hub:
                self.model, self.target_size = mpp.create_model(model_type=self.model_type,
                                                                load_weights=False,
                                                                file_of_weight=file_of_weight);
            
            elif self.model_type in self.model_list_yolo:
                # Warning: This code was put here because load torch that has 8.5.0 cudnn version, but tensorflow have 8.9.* version
                import BodyEmotion4Lib.lib_yolo_model as myp
                self.model, self.target_size = myp.create_model(model_type=self.model_type,
                                                                load_weights=False,
                                                                file_of_weight=file_of_weight);
            
            else:
                print("Don't exist the model:", self.model_type);
                exit();
        
        else:
            if   self.model_type in self.model_list_hub:
                self.model, self.target_size=mpp.create_model(  model_type=self.model_type,
                                                                load_weights=True,
                                                                file_of_weight='');
                
            elif self.model_type in self.model_list_yolo:
                # Warning: This code was put here because load torch that has 8.5.0 cudnn version, but tensorflow have 8.9.* version
                import BodyEmotion4Lib.lib_yolo_model as myp
                self.model, self.target_size=myp.create_model(  model_type=self.model_type,
                                                                load_weights=True,
                                                                file_of_weight='');
                
            else:
                print("Don't exist the model:", self.model_type);
                exit();
        
    
    def from_img_filepath(self,imgfilepath):
        """Classify a image from the image filepath.
        
        Args:
            imgfilepath: The iamge filepath.
        
        Returns:
            int: The class of image.
        """
        if   self.model_type in self.model_list_hub:
            return mpp.evaluate_model_from_file(self.model,imgfilepath, target_size=self.target_size);
        elif self.model_type in self.model_list_yolo:
            # Warning: This code was put here because load torch that has 8.5.0 cudnn version, but tensorflow have 8.9.* version
            import BodyEmotion4Lib.lib_yolo_model as myp
            return myp.evaluate_model_from_file(self.model,imgfilepath);
        else:
            return 0;

    def from_img_pil(self,img_pil):
        """Classify a image from a PIL object.
        
        Args:
            img_pil: The PIL object.
        
        Returns:
            int: The class of image.
        """
        if   self.model_type in self.model_list_hub:
            return mpp.evaluate_model_from_pil(self.model,img_pil, target_size=self.target_size);
        elif self.model_type in self.model_list_yolo:
            # Warning: This code was put here because load torch that has 8.5.0 cudnn version, but tensorflow have 8.9.* version
            import BodyEmotion4Lib.lib_yolo_model as myp
            return myp.evaluate_model_from_pil(self.model,img_pil);
        else:
            return 0;

    def predict_pil(self,img_pil):
        """Classify a image from a PIL object.
        
        Args:
            img_pil: The PIL object.
        
        Returns:
            int: The class of image.
        """
        if   self.model_type in self.model_list_hub:
            return mpp.predict_from_pil(self.model,img_pil, target_size=self.target_size);
        elif self.model_type in self.model_list_yolo:
            # Warning: This code was put here because load torch that has 8.5.0 cudnn version, but tensorflow have 8.9.* version
            import BodyEmotion4Lib.lib_yolo_model as myp
            return myp.predict_from_pil(self.model,img_pil);
        else:
            return 0;

    def target_labels(self):
        """Returns the categories of classifier.
        
        Returns:
            list: The labels of categories resturned by the methods from_img_pil() and from_img_filepath().
        """
        return ['negative','neutral','pain','positive'];


