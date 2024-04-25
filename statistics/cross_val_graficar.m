%
dpath{1} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_resnet_v2_50_acc9286';
dpath{2} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_mobilenet_v3_acc9144';
dpath{3} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_inception_resnet_v2_acc9381';
dpath{4} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_inception_v3_acc9191';
dpath{5} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_efficientnet_b3_acc9397';

model_type{1}='resnet\_v2\_50';
model_type{2}='mobilenet\_v3';
model_type{3}='inception\_resnet\_v2';
model_type{4}='inception\_v3';
model_type{5}='efficientnet\_b3';

L=length(dpath);

trainparams=zeros(L,1);
totalparams=zeros(L,1);
acc_val=zeros(L,1);

for II=1:L
    source(fullfile(dpath{II},'parameters_stats.m'))
    trainparams(II)=parameters_trainable;
    totalparams(II)=parameters_total;
    
    source(fullfile(dpath{II},'final_stats.m'))
    acc_val(II)=mean_val_acc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FONTSIZE=12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
hf=figure(1);
scatter (trainparams, acc_val, 1200*(totalparams)/max(totalparams),1:L,'filled')
hx=xlabel('# of trainable parameters');
hy=ylabel('Mean of validation accuracy');
set(hx,'fontsize',FONTSIZE);
set(hy,'fontsize',FONTSIZE);
set(gca,'fontsize',FONTSIZE);

for II=1:L
    text (trainparams(II)*1.03, acc_val(II), model_type{II},'fontsize',FONTSIZE)
end

print(gcf,'cross_val_acc_val_vs_trainable.eps','-depsc')
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hf=figure(2);
scatter (totalparams, acc_val, 1200*(totalparams)/max(totalparams),1:L,'filled')
hx=xlabel('# of parameters');
hy=ylabel('Mean of validation accuracy');
set(hx,'fontsize',FONTSIZE);
set(hy,'fontsize',FONTSIZE);
set(gca,'fontsize',FONTSIZE);

for II=1:L
    text (totalparams(II)*0.88, 0.997*acc_val(II), model_type{II},'fontsize',FONTSIZE)
end

print(gcf,'cross_val_acc_val_vs_total.eps','-depsc')
