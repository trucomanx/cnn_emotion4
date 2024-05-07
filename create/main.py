
################################################################################
################################################################################
################################################################################

import os
import json
import numpy as np
import sys

for n in range(len(sys.argv)):
    if sys.argv[n]=='--json-file':
        json_filename=sys.argv[n+1];

testing=dict();
for info in info_list:
    testing[info]=[];

base_name=os.path.splitext(os.path.basename(json_filename))[0]

fout = open(base_name+"_summary.csv", "w")


fout.write('Model'+sep+ sep.join(info_list)+'\n');
for model in model_list:
    # Opening JSON file
    f = open(os.path.join(model,json_filename))
     
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    row=[];
    for info in info_list:
        testing[info].append(data[info]);
        row.append(data[info]);
    # writing
    fout.write( model+sep+sep.join([str(x) for x in row])+'\n' )
    
    # Closing file
    f.close()

fout.close()




# importing matplotlib
import matplotlib.pyplot as plt
import matplotlib

for info in info_list:
    plt.figure(figsize=(15,6))
    matplotlib.rcParams.update({'font.size': 18})

    plt.bar(model_list, testing[info])

    for n in range(len(model_list)):
        plt.text(model_list[n], testing[info][n]+0.005, round(testing[info][n],3),fontsize=16)

    plt.title(info)
    plt.ylim(np.min(testing[info])/1.1, np.max(testing[info])*1.1) 
    plt.grid(True) 

    plt.savefig(base_name+'_'+info+image_ext);

    #plt.show()

if 'erro_bar' in locals():
    for item in erro_bar:
        plt.figure(figsize=(15,6))
        matplotlib.rcParams.update({'font.size': 18})

        plt.bar(model_list, testing[item[0]], yerr=testing[item[1]], capsize=24) 
        
        plt.title(item[0]+' , '+item[1])
        plt.ylim(np.min(np.array(testing[item[0]])-np.array(testing[item[1]]))/1.1, np.max(np.array(testing[item[0]])+np.array(testing[item[1]]))*1.1) 
        plt.grid(True) 

        plt.savefig(base_name+'_error_'+info+image_ext);
            
