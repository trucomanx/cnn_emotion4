# %%
input_default_json_conf_file='cnn_emotion4_kfold_default.json';

# %%
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import datetime
import json

# %%
import sys
sys.path.append('../library');

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
## Load json conf json file
fd = open(os.path.join('./',input_default_json_conf_file));
DATA = json.load(fd);
fd.close()

# %%
"""
# Variable globales
"""

# %%
## Seed for the random variables
seed_number = 0;

## Dataset 
dataset_base_dir    = DATA["dataset_base_dir"]; 
dataset_labels_file = DATA['dataset_labels_file'];
dataset_name        = DATA['dataset_name'];

## Kfold 
K                 = DATA["kfold"]; # Variable K of kfold
enable_stratified = DATA["enable_stratified"]; # True: Stratified kfold False: Enable kfold 

## Training hyperparameters
EPOCAS     = DATA["epochs"];
BATCH_SIZE = DATA["batch_size"];

## Model of network
## 'mobilenet_v3', 'efficientnet_b3', 'inception_v3', 'inception_resnet_v2', 'resnet_v2_50'
model_type = DATA["model_type"];

## Output
output_base_dir = DATA["output_base_dir"];

## Output json file
fold_status_file='kfold_data_results.json';

##
full_learning = DATA["full_learning"];

##############################################

#print('   dataset_base_dir:',dataset_base_dir)
#print('dataset_labels_file:',dataset_labels_file)
#print('       dataset_name:',dataset_name)
#print('         model_type:',model_type)
#print('                  K:',K)
#print('  enable_stratified:',enable_stratified)
#print('             EPOCAS:',EPOCAS)
#print('         BATCH_SIZE:',BATCH_SIZE)
#print('    output_base_dir:',output_base_dir)


# %%
"""
# Parametros de entrada
"""

# %%
for n in range(len(sys.argv)):
    if sys.argv[n]=='--dataset-dir':
        dataset_base_dir=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-file':
        dataset_labels_file=sys.argv[n+1];
    elif sys.argv[n]=='--dataset-name':
        dataset_name=sys.argv[n+1];
    elif sys.argv[n]=='--model':
        model_type=sys.argv[n+1];
    elif sys.argv[n]=='--kfold':
        K=int(sys.argv[n+1]);
    elif sys.argv[n]=='--enable-stratified':
        enable_stratified=bool(sys.argv[n+1]);
    elif sys.argv[n]=='--epochs':
        EPOCAS=int(sys.argv[n+1]);
    elif sys.argv[n]=='--batch-size':
        BATCH_SIZE=int(sys.argv[n+1]);
    elif sys.argv[n]=='--output-dir':
        output_base_dir=sys.argv[n+1];
    elif sys.argv[n]=='--full-learning':
        full_learning=sys.argv[n+1].lower()=='true';
        
print('   dataset_base_dir:',dataset_base_dir)
print('dataset_labels_file:',dataset_labels_file)
print('       dataset_name:',dataset_name)
print('         model_type:',model_type)
print('                  K:',K)
print('  enable_stratified:',enable_stratified)
print('             EPOCAS:',EPOCAS)
print('         BATCH_SIZE:',BATCH_SIZE)
print('    output_base_dir:',output_base_dir)
print('      full_learning:',full_learning)

# %%
"""
# Set seed of random variables

"""

# %%
np.random.seed(seed_number)
tf.keras.utils.set_random_seed(seed_number);

# %%
"""
# Setting the cross-validation kfold

"""

# %%
from sklearn.model_selection import KFold, StratifiedKFold

if full_learning:
    output_dir = os.path.join(output_base_dir,dataset_name,'cross-validation_full-learning',model_type);
else:
    output_dir = os.path.join(output_base_dir,dataset_name,'cross-validation',model_type);

if enable_stratified:
    kf = StratifiedKFold(n_splits = K, shuffle = True, random_state = seed_number);
else:
    kf  = KFold(n_splits = K, shuffle=True, random_state=seed_number); 

# %%
"""
# Loading data of dataset
"""

# %%
# Load filenames and labels
train_val_data = pd.read_csv(os.path.join(dataset_base_dir,dataset_labels_file));
print(train_val_data)
# Setting labels
Y   = train_val_data[['label']];
L=np.shape(Y)[0];

# %%
"""
# Data augmentation configuration
"""

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg    = ImageDataGenerator(rescale=1./255,
                            rotation_range = 10,
                            width_shift_range= 0.07,
                            height_shift_range= 0.07,
                            horizontal_flip=True,
                            shear_range=1.25,
                            zoom_range = [0.75, 1.25] 
                            )

idg_val= ImageDataGenerator(rescale=1./255 )



# %%
"""
# Auxiliar function
"""

# %%
def get_model_name(k):
    return 'model_'+str(k)+'.h5'

# %%
"""
# Creating output directory
"""

# %%
os.makedirs(output_dir,exist_ok = True) 
print(output_dir)


# %%
"""
# Creating output status file
"""

# %%
json_status_path=os.path.join(output_dir,fold_status_file);

# %%
"""
# Cross-validation
"""

# %%
import BodyEmotion4Lib.lib_model as mpp
import matplotlib.pyplot as plt

list_train_index=[];
list_val_index=[];
for train_index, val_index in kf.split(np.zeros(L),Y):
    list_train_index.append(train_index);
    #print('train_index:',train_index);
    list_val_index.append(val_index);
    #print('val_index:',val_index);

data_fold =  {'val_categorical_accuracy': [],'val_loss': [], 'train_categorical_accuracy': [],'train_loss': [] };

fold_var=1;

if os.path.isfile(json_status_path):
    # Read JSON file
    with open(json_status_path) as data_file:
        data_fold = json.load(data_file)
        fold_var=len(data_fold['val_loss'])+1;

while fold_var<=K:
    training_data   = train_val_data.iloc[list_train_index[fold_var-1]]
    validation_data = train_val_data.iloc[list_val_index[fold_var-1]]

    print('\nFold:',fold_var);
    
    # CREATE NEW MODEL
    model, target_size = mpp.create_model(model_type=model_type,load_weights=False);
    if full_learning:
        for layer in model.layers:
            layer.trainable = True;
    model.summary()

    train_data_generator = idg.flow_from_dataframe(training_data, 
                                                   directory = dataset_base_dir,
                                                   target_size=target_size,
                                                   x_col = "filename", 
                                                   y_col = "label",
                                                   batch_size=BATCH_SIZE,
                                                   class_mode="categorical",
                                                   shuffle = True);
    
    valid_data_generator  = idg_val.flow_from_dataframe(validation_data, 
                                                    directory = dataset_base_dir,
                                                    target_size=target_size,
                                                    x_col = "filename", 
                                                    y_col = "label",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    shuffle = True)
    
    STEPS_BY_EPOCHS=len(train_data_generator);
    

    
    # COMPILE NEW MODEL
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    
    # CREATE CALLBACKS
    best_model_file=os.path.join(output_dir,get_model_name(fold_var));
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file, 
                                                    save_weights_only=True,
                                                    monitor='val_loss', 
                                                    save_best_only=True, 
                                                    verbose=1);
    
    # Definindo o callback EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(  monitor='val_loss', 
                                                        patience=max(10,int(EPOCAS/5)),  
                                                        verbose=1, 
                                                        restore_best_weights=False);

    log_dir = os.path.join(output_dir,"logs","fit", "fold"+str(fold_var)+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"));
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL
    history = model.fit(train_data_generator,
                        steps_per_epoch=STEPS_BY_EPOCHS,
                        epochs=EPOCAS,
                        validation_data=valid_data_generator,
                        callbacks=[checkpoint,early_stopping,tensorboard_callback],
                        verbose=1
                       );
    
    #PLOT HISTORY
    mpp.save_model_history(history,
                           os.path.join(output_dir,"historical_"+str(fold_var)+".csv"),
                           show=False,
                           labels=['categorical_accuracy','loss']);
    
    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights(best_model_file);
    
    # Evaluate training
    results = model.evaluate(train_data_generator);
    results = dict(zip(model.metrics_names,results));
    print("Training:\n",results,"\n\n");
    data_fold['train_categorical_accuracy'].append(results['categorical_accuracy']);
    data_fold['train_loss'].append(results['loss']);

    # Evaluate validation
    results = model.evaluate(valid_data_generator)
    results = dict(zip(model.metrics_names,results))
    print("Validation:\n",results,"\n\n");
    data_fold['val_categorical_accuracy'].append(results['categorical_accuracy'])
    data_fold['val_loss'].append(results['loss'])

    # Data fold
    with open(json_status_path, 'w') as f:
        json.dump(data_fold, f,indent=4);

    tf.keras.backend.clear_session()
    
    fold_var += 1

# %%
import json


data_fold['mean_val_categorical_accuracy'] = np.mean(data_fold['val_categorical_accuracy']);
data_fold['std_val_categorical_accuracy']  = np.std(data_fold['val_categorical_accuracy']);

data_fold['mean_val_loss'] = np.mean(data_fold['val_loss']);
data_fold['std_val_loss']  = np.std(data_fold['val_loss']);

data_fold['mean_train_categorical_accuracy'] = np.mean(data_fold['train_categorical_accuracy']);
data_fold['std_train_categorical_accuracy']  = np.std(data_fold['train_categorical_accuracy']);

data_fold['mean_train_loss'] = np.mean(data_fold['train_loss']);
data_fold['std_train_loss']  = np.std(data_fold['train_loss']);

print(data_fold)

# Data fold
with open(json_status_path, 'w') as f:
    json.dump(data_fold, f,indent=4);

# %%
fpath=os.path.join(output_dir,"final_stats.m");
mean_val_acc=mpp.save_model_stat_kfold(data_fold['val_categorical_accuracy'],data_fold['val_loss'], fpath);

mpp.save_model_parameters(model, os.path.join(output_dir,'parameters_stats.m'));

print(mean_val_acc)