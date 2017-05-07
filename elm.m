clc;
clear;
%%%%%%隐层神经元数设置
NumberofHiddenNeurons = 20;
%%%%%%载入training set.
load('Eng_H_0_Ma_0.mat');
T = y_te';                   %训练集的第1列为输出
P = x_te';

clear train_data;


NumberofTrainingData = size(P,2);
NumberofInputNeurons = size(P,1);



%%%%%%%随机生成输入权重InputWeight(w_i)和偏置 BiasofHiddenNeurons(b_i)
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons) * 2 - 1;
BiasofHiddenNeurons = rand(NumberofHiddenNeurons,1);
tempH=InputWeight * P;
clear P;
ind = ones(1,NumberofTrainingData);
BiasMatrix = BiasofHiddenNeurons(:,ind);       %使矩阵满足H的维度
tempH = tempH + BiasMatrix;

%%%%%%%计算输出矩阵H
%%%%激活函数选择
H=radbas(tempH);                     %RBF

%%%%%计算输出权重 OutputWeight (beata_i)
OutputWeight = pinv(H') * T';    %广义逆，without regularization


%%%%%% 计算训练准确度
Y = (H' * OutputWeight)';
TrainingAccuracy = sqrt(mse(T - Y))      %计算RMSE

plot(Y);
hold on
plot(y_tr)
hold off
