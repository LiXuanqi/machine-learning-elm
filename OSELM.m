clc;
clear;
%%%%%%��ʼ��������
NumberofHiddenNeurons = 20;    %������Ԫ��
N0 = 100;                        %���ڳ�ʼ���׶ε���������
Block = 20;                      %����ѧϰ ÿ����������
%%%%%%����training set.
train_data = load('housing.txt');
T = train_data(:,14);                   %ѵ�����ĵ�14��Ϊ���
P = train_data(:,1:13);

clear train_data;

%%%%%������Լ�����ʲô��

NumberofTrainingData = size(P,1);
NumberofInputNeurons = size(P,2);

%%%%% ��ʼ���׶�
P0 = P(1:N0,:);           %ȡǰN0������
T0 = T(1:N0,:);
%%%%%%%�����������Ȩ��InputWeight(w_i)��ƫ�� BiasofHiddenNeurons(b_i)
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons) * 2 - 1;

BiasofHiddenNeurons = rand(1,NumberofHiddenNeurons);    %��ͬ�����ƫ��ѡ��ͬ



H0 = RBFun(P0,InputWeight,BiasofHiddenNeurons);    %RBF�����

M = pinv(H0' * H0);      %��K0�Ĺ�����
beta = pinv(H0) * T0;     %��beta_0

clear P0 H0;
%%%%%%%%%% ����ѧϰ�׶�
for n = N0 : Block : NumberofTrainingData
    if (n + Block - 1) > NumberofTrainingData
        Pn = P(n:NumberofTrainingData,:);
        Tn = T(NumberofTrainingData,:);
        Block = size(Pn,1);
        clear V;
    else
        Pn = P(n:(n+Block-1),:);
        Tn = T(n:(n+Block-1),:);
    end
    H = RBFun(Pn,InputWeight,BiasofHiddenNeurons);
    
    M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M; %M�ĵ�����ʽ
    beta = beta + M * H' * (Tn - H * beta);   %beta�ĵ�����ʽ

end


HTrain = RBFun(P,InputWeight,BiasofHiddenNeurons);

Y = HTrain * beta;
clear HTrain;


%%%%%% ����ѵ��׼ȷ��

TrainingAccuracy = sqrt(mse(T - Y))      %����RMSE

