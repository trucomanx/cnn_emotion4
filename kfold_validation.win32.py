#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf
import datetime


# In[2]:


import sys
sys.path.append("library");


# # Variable globales

# In[3]:


## Seed for the random variables
seed_number=0;

## Dataset 
dataset_base_dir    = "../../dataset/patient_people/train";
dataset_labels_file = "emotion4.work1.csv";

## Kfold 
K=5;                    # Variable K of kfold
enable_stratified=True; # True: Stratified kfold False: Enable kfold 

## Training hyperparameters
EPOCAS=50;
BATCH_SIZE=32;

## Model of network
#model_type  = "mobilenet_v3";
#model_type = "efficientnet_b3"
#model_type = "inception_v3";
#model_type = "inception_resnet_v2";
model_type = "resnet_v2_50";

## Output
output_base_dir = "D:\FernandoPujaicoRivera\output\cnn_emotion4\cross-validation";


## Parametros de entrada
for n in range(len(sys.argv)):
    if sys.argv[n]=="--model":
        model_type=sys.argv[n+1];

print("model_type",model_type)
# # Set seed of random variables
# 

# In[4]:


np.random.seed(seed_number)
tf.keras.utils.set_random_seed(seed_number);


# # Setting the cross-validation kfold
# 

# In[5]:


from sklearn.model_selection import KFold, StratifiedKFold

if enable_stratified:
    output_dir = os.path.join(output_base_dir,"skfold");
    kf = StratifiedKFold(n_splits = K, shuffle = True, random_state = seed_number);
else:
    output_dir = os.path.join(output_base_dir,"kfold");
    kf  = KFold(n_splits = K, shuffle=True, random_state=seed_number); 


# # Loading data of dataset

# In[6]:


# Load filenames and labels
train_data = pd.read_csv(os.path.join(dataset_base_dir,dataset_labels_file));
print(train_data)
# Setting labels
Y   = train_data[["label"]];
L=np.shape(Y)[0];


# # Data augmentation configuration

# In[7]:


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



# # Auxiliar function

# In[8]:


def get_model_name(k):
    return "model_"+str(k)+".h5"


# # Creating output directory

# In[9]:


try: 
    os.mkdir(output_dir) 
except: 
    pass


# # Cross-validation

# In[ ]:


import lib_model as mpp
import matplotlib.pyplot as plt

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []


fold_var=1;
for train_index, val_index in kf.split(np.zeros(L),Y):
    training_data   = train_data.iloc[train_index]
    validation_data = train_data.iloc[val_index]

    print("\nFold:",fold_var);
    
    # CREATE NEW MODEL
    model, target_size = mpp.create_model("",model_type=model_type);
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
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    
    # CREATE CALLBACKS
    best_model_file=os.path.join(output_dir,get_model_name(fold_var));
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_file, 
                                                    save_weights_only=True,
                                                    monitor="val_accuracy", 
                                                    save_best_only=True, 
                                                    verbose=1);
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL
    history = model.fit(train_data_generator,
                        steps_per_epoch=STEPS_BY_EPOCHS,
                        epochs=EPOCAS,
                        validation_data=valid_data_generator,
                        callbacks=[checkpoint,tensorboard_callback],
                        verbose=1
                       );
    
    #PLOT HISTORY
    mpp.save_model_history(history,os.path.join(output_dir,"historical_"+str(fold_var)+".csv"),show=False);
    
    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights(best_model_file);
    
    results = model.evaluate(valid_data_generator)
    results = dict(zip(model.metrics_names,results))
    print(results,"\n\n");
    
    VALIDATION_ACCURACY.append(results["accuracy"])
    VALIDATION_LOSS.append(results["loss"])
    
    tf.keras.backend.clear_session()
    
    fold_var += 1


# In[ ]:


fpath=os.path.join(output_dir,"final_stats.m");
mean_val_acc=mpp.save_model_stat_kfold(VALIDATION_ACCURACY,VALIDATION_LOSS, fpath);

mpp.save_model_parameters(model, os.path.join(output_dir,"parameters_stats.m"));

os.rename(output_dir,output_dir+str(K)+"_"+model_type+"_acc"+str(int(mean_val_acc*10000)));
print(mean_val_acc)


# In[ ]:





# In[ ]:




