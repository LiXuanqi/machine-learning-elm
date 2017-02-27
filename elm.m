clc;
clear;
%%%%%%隐层神经元数设置
NumberofHiddenNeurons = 20;
%%%%%%载入training set.
train_data = load('housing.txt');
T = train_data(:,14)';                   %训练集的第14列为输出
P = train_data(:,1:13)';

clear train_data;

%%%%%载入测试集，有什么用

NumberofTrainingData = size(P,2);
NumberofInputNeurons = size(P,1);

%%%%%%%training set 预处理
temp_T = zeros(1,NumberofTrainingData);

T = temp_T * 2 - 1;                      %涵义？

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

