clc;
clear;
%%%%%%������Ԫ������
NumberofHiddenNeurons = 20;
%%%%%%����training set.
train_data = load('housing.txt');
T = train_data(:,14)';                   %ѵ�����ĵ�14��Ϊ���
P = train_data(:,1:13)';

clear train_data;

%%%%%������Լ�����ʲô��

NumberofTrainingData = size(P,2);
NumberofInputNeurons = size(P,1);

%%%%%%%training set Ԥ����
temp_T = zeros(1,NumberofTrainingData);

T = temp_T * 2 - 1;                      %���壿

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

