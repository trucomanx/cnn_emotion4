import os
import sys
from ultralytics import YOLO

def create_model(model_type='yolov8n-cls',load_weights=True,file_of_weight=''):
    
    model_list_yolo = ['yolov8n-cls','yolov8s-cls','yolov8m-cls'];
    
    if   model_type in model_list_yolo:
        model = YOLO(model_type+'.yaml').load(model_type+'.pt');
        target_size=(224,224);
    else:
        raise TypeError("Unknown parameter model_type");
        
    print("Loading architecture",model_type);
    
    print('');
    print('        url:',model_type+'.yaml');
    print('target_size:',target_size);
    print('');
        
    if load_weights==True:
        path_actual = os.path.realpath(__file__);
        directorio_actual = os.path.dirname(path_actual);
        path_of_model=os.path.join(directorio_actual,'models','model_'+model_type+'.pt');
        
        if os.path.exists(path_of_model):
            print("Loading the weights in:",path_of_model);
            try:
                model=model.load(path_of_model);
                print("Loaded the weights in:",path_of_model);
            except Exception:
                print("Error loading the weights in:",path_of_model);
                exit();
        else:
            print("Error loading, file no found:",path_of_model);
    
    if len(file_of_weight)!=0:
        print("Loading the weights in:",file_of_weight);
        if os.path.exists(file_of_weight):
            #
            try:
                model=model.load(file_of_weight);
                print("Loaded the weights in:",file_of_weight);
            except Exception:
                print("Error loading the weights in:",file_of_weight);
                exit();
        else:
            print("Error loading, file no found:",file_of_weight);
    
    return model, target_size;

def evaluate_model_from_file(model, imgfilepath):
    '''
    Evalua la red neuronal descrita en `model`, la entrada es leida desde el archivo `imgfilepath`.
    
    '''
    
    # https://docs.ultralytics.com/modes/predict/#__tabbed_2_4
    res=model(imgfilepath).probs.top1;
    
    return res;

def evaluate_model_from_pil(model, image):
    '''
    Evalua la red neuronal descrita en `model`, la entrada es leida desde una imagen PIL.
    
    '''
    
    # https://docs.ultralytics.com/modes/predict/#__tabbed_2_4
    res=model(image).probs.top1;

    return res;

