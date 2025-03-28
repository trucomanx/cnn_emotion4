import os
import sys
from ultralytics import YOLO

def create_model(model_type='yolov8n-cls',load_weights=True,file_of_weight=''):
    
    model_list_yolo = ['yolov8n-cls','yolov8s-cls','yolov8m-cls'];
    
    def loading_default(model_type,model_list_yolo):
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
        
        return model,target_size;
    
    model=None;
    target_size=None;
    
    if load_weights==True:
        path_actual = os.path.realpath(__file__);
        directorio_actual = os.path.dirname(path_actual);
        path_of_model=os.path.join(directorio_actual,'models','model_'+model_type+'.pt');
        
        if os.path.exists(path_of_model):
            print("Loading the weights in:",path_of_model);
            try:
                model=YOLO(path_of_model);
                target_size=(224,224);
                print("Loaded the weights in:",path_of_model);
            except Exception:
                print("Error loading the weights in:",path_of_model);
                exit();
        else:
            print("Error loading, file no found:",path_of_model);
            exit();

    
    if len(file_of_weight)!=0:
        print("Loading the weights in:",file_of_weight);
        if os.path.exists(file_of_weight):
            #
            try:
                model=YOLO(file_of_weight);
                target_size=(224,224);
                print("Loaded the weights in:",file_of_weight);
            except Exception:
                print("Error loading the weights in:",file_of_weight);
                exit();
        else:
            print("Error loading, file no found:",file_of_weight);
            exit();
    
    if model==None or target_size==None:
        model, target_size=loading_default(model_type,model_list_yolo);
    
    return model, target_size;

def evaluate_model_from_file(model, imgfilepath):
    '''
    Evalua la red neuronal descrita en `model`, la entrada es leida desde el archivo `imgfilepath`.
    
    '''
    
    # https://docs.ultralytics.com/modes/predict/#__tabbed_2_4
    res=model(imgfilepath)[0].probs.top1;
    
    return res;

def evaluate_model_from_pil(model, image):
    '''
    Evalua la red neuronal descrita en `model`, la entrada es leida desde una imagen PIL.
    
    '''
    
    # https://docs.ultralytics.com/modes/predict/#__tabbed_2_4
    res=model(image)[0].probs.top1;

    return res;
    
def predict_from_pil(model, image):
    '''
    Evalua la red neuronal descrita en `model`, la entrada es leida desde una imagen PIL.
    
    '''
    
    # https://docs.ultralytics.com/modes/predict/#__tabbed_2_4
    res=model(image)[0].probs;

    return res;

def get_model_parameters(model):
    return model.info()[1];
    
def save_model_parameters(model, fpath):
    '''
    Salva en un archivo la estadistica de la cantidoda de parametros de un modelo
    
    :param model: Modelos a analizar
    :type model: str
    :param fpath: Archivo donde se salvaran los datos.
    :type fpath: str
    '''
    trainable_count = model.info()[1];
    
    fid = open(fpath, 'w')
    print('parameters_total={}'.format(trainable_count),';', file = fid);
    print('parameters_trainable={}'.format(trainable_count),';', file = fid);
    fid.close()
    
