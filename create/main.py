
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


def load_kfold_data(json_filename,model_list,sort_by):
    sample=dict();
    for model in model_list:
        # Opening JSON file
        f = open(os.path.join(model,json_filename))
         
        # returns JSON object as 
        # a dictionary
        data = json.load(f);
        sample[model]=data[sort_by];
    return sample;

if 'sort_by' in locals():
    from functools import cmp_to_key
    from scipy import stats
    
    sample = load_kfold_data(json_filename,model_list,sort_by);
    
    def eh_maior(a, b, p_max=0.05):
        # Realizando o teste t de Student para amostras independentes
        t_stat_1, p_value_1 = stats.ttest_ind(sample[a], sample[b],alternative='greater')
        
        # Realizando o teste t de Student para amostras independentes
        t_stat_2, p_value_2 = stats.ttest_ind(sample[b], sample[a],alternative='greater')
        
        if   p_value_1 < p_max:
            return 1;
        elif p_value_2 < p_max:
            return -1
        else:
            return 0

    # Função de chave usando cmp_to_key
    key_func = cmp_to_key(eh_maior);

    # Ordenando a lista
    model_list = sorted(model_list, key=key_func, reverse=True)

# creating testing dict
testing=dict();
for info in info_list:
    testing[info]=[];

# other
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

        plt.savefig(base_name+'_error_'+item[0]+image_ext);

if 'p_matrix' in locals():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    s_data = load_kfold_data(json_filename,model_list,p_matrix);
    
    def a_maior_b(a, b, s_data):
        # Realizando o teste t de Student para amostras independentes
        t_stat, p_value = stats.ttest_ind(s_data[a], s_data[b],alternative='greater')
        
        return t_stat, p_value;
    # Inicializar a matriz de confusão de p-valores
    p_values = np.zeros((len(model_list), len(model_list)))

    # Preencher a matriz de confusão com p-valores
    for i in range(len(model_list)):
        for j in range(len(model_list)):
            if i < j:
                t_stat, p_val = a_maior_b(model_list[i], model_list[j], s_data)
                p_values[i, j] = p_val;
            else:
                p_values[i, j] = np.nan  # Diagonal principal será NaN
    
    
    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 8))
    ax=sns.heatmap(p_values, annot=True, fmt='.2f', cmap="coolwarm", xticklabels=model_list, yticklabels=model_list, mask=np.isnan(p_values))
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    # Rotacionar e ajustar os rótulos
    #plt.xticks(rotation=45, ha='right', fontsize=10)
    #plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.title('p-values A > B')
    plt.xlabel('B')
    plt.ylabel('A')
    plt.savefig(base_name+'_p_val_'+p_matrix+image_ext);
    #plt.show();
