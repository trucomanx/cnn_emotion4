%
dpath{1} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test/resnet_v2_50';
dpath{2} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test/mobilenet_v3';
dpath{3} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test/inception_resnet_v2';
dpath{4} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test/inception_v3';
dpath{5} = '/mnt/boveda/DOCTORADO2/cnn_patient_people/test/efficientnet_b3';


model_type{1}='resnet\_v2\_50';
model_type{2}='mobilenet\_v3';
model_type{3}='inception\_resnet\_v2';
model_type{4}='inception\_v3';
model_type{5}='efficientnet\_b3';

L=length(dpath);

delay_ms=zeros(L,1);
acc_test=zeros(L,1);

trainparams=zeros(L,1);
totalparams=zeros(L,1);

for II=1:L
    source(fullfile(dpath{II},'results_testing.m'))
    delay_ms(II)=delayms;
    acc_test(II)=accuracy;
    
    source(fullfile(dpath{II},'parameters_stats.m'))
    trainparams(II)=parameters_trainable;
    totalparams(II)=parameters_total;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FONTSIZE=12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hf=figure(1);
scatter (delay_ms, acc_test, 1200*(totalparams)/max(totalparams),1:L,'filled')
hx=xlabel('Delay in ms');
hy=ylabel('Test custom accuracy');
%ylim([0.92 0.98])
set(hx,'fontsize',FONTSIZE);
set(hy,'fontsize',FONTSIZE);
set(gca,'fontsize',FONTSIZE);

randn('seed', 11)
for II=1:L
    offset=0;
    if (II==4)
        offset=0.003;
    end
    text (delay_ms(II)*0.7, offset+acc_test(II)*(1+0.005*randn(1)), model_type{II},'fontsize',FONTSIZE)
end

print(gcf,'test_holdout_acc_vs_delayms.eps','-depsc')

