%
dpath{1} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test_custom/delay_resnet_v2_50';
dpath{2} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test_custom/delay_mobilenet_v3';
dpath{3} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test_custom/delay_inception_resnet_v2';
dpath{4} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test_custom/delay_inception_v3';
dpath{5} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test_custom/delay_efficientnet_b3';

%
dpathP{1} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_resnet_v2_50_acc9286';
dpathP{2} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_mobilenet_v3_acc9144';
dpathP{3} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_inception_resnet_v2_acc9381';
dpathP{4} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_inception_v3_acc9191';
dpathP{5} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/cross-validation/skfold5_efficientnet_b3_acc9397';

model_type{1}='resnet\_v2\_50';
model_type{2}='mobilenet\_v3';
model_type{3}='inception\_resnet\_v2';
model_type{4}='inception\_v3';
model_type{5}='efficientnet\_b3';

L=length(dpath);

delay_ms=zeros(L,1);
acc_testc=zeros(L,1);

trainparams=zeros(L,1);
totalparams=zeros(L,1);

for II=1:L
    source(fullfile(dpath{II},'times10_acc_delayms.m'))
    delay_ms(II)=delayms;
    acc_testc(II)=acc;
    
    source(fullfile(dpathP{II},'parameters_stats.m'))
    trainparams(II)=parameters_trainable;
    totalparams(II)=parameters_total;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FONTSIZE=12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hf=figure(1);
scatter (delay_ms, acc_testc, 1200*(totalparams)/max(totalparams),1:L,'filled')
hx=xlabel('Delay in ms');
hy=ylabel('Test custom accuracy');
ylim([0.92 0.98])
set(hx,'fontsize',FONTSIZE);
set(hy,'fontsize',FONTSIZE);
set(gca,'fontsize',FONTSIZE);

randn('seed', 11)
for II=1:L
    offset=0;
    if (II==4)
        offset=0.003;
    end
    text (delay_ms(II)*0.90, offset+acc_testc(II)*(1+0.005*randn(1)), model_type{II},'fontsize',FONTSIZE)
end

print(gcf,'test_custom_acc_vs_delayms.eps','-depsc')

