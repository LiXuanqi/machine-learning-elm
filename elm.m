clc;
clear;
%%%%%%������Ԫ������
NumberofHiddenNeurons = 20;
%%%%%%����training set.
load('Eng_H_0_Ma_0.mat');
T = y_te';                   %ѵ�����ĵ�1��Ϊ���
P = x_te';

clear train_data;


NumberofTrainingData = size(P,2);
NumberofInputNeurons = size(P,1);



%%%%%%%�����������Ȩ��InputWeight(w_i)��ƫ�� BiasofHiddenNeurons(b_i)
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons) * 2 - 1;
BiasofHiddenNeurons = rand(NumberofHiddenNeurons,1);
tempH=InputWeight * P;
clear P;
ind = ones(1,NumberofTrainingData);
BiasMatrix = BiasofHiddenNeurons(:,ind);       %ʹ��������H��ά��
tempH = tempH + BiasMatrix;

%%%%%%%�����������H
%%%%�����ѡ��
H=radbas(tempH);                     %RBF

%%%%%�������Ȩ�� OutputWeight (beata_i)
OutputWeight = pinv(H') * T';    %�����棬without regularization


%%%%%% ����ѵ��׼ȷ��
Y = (H' * OutputWeight)';
TrainingAccuracy = sqrt(mse(T - Y))      %����RMSE

plot(Y);
hold on
plot(y_tr)
hold off
