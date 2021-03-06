clc;
clear;
%%%%%%初始参数设置
NumberofHiddenNeurons = 20;    %隐层神经元数
N0 = 100;                        %用于初始化阶段的样本数量
Block = 20;                      %在线学习 每步的数据量
%%%%%%载入training set.
train_data = load('sinc_train');
T = train_data(:,1);                   %训练集的第14列为输出
P = train_data(:,2);

clear train_data;

%%%%%载入测试集，有什么用

NumberofTrainingData = size(P,1);
NumberofInputNeurons = size(P,2);

%%%%% 初始化阶段
P0 = P(1:N0,:);           %取前N0行数据
T0 = T(1:N0,:);
%%%%%%%随机生成输入权重InputWeight(w_i)和偏置 BiasofHiddenNeurons(b_i)
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons) * 2 - 1;

BiasofHiddenNeurons = rand(1,NumberofHiddenNeurons);    %不同激活函数偏置选择不同



H0 = RBFun(P0,InputWeight,BiasofHiddenNeurons);    %RBF激活函数

M = pinv(H0' * H0);      %求K0的广义逆
beta = pinv(H0) * T0;     %求beta_0

clear P0 H0;
%%%%%%%%%% 在线学习阶段
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
    
    M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M; %M的迭代公式
    beta = beta + M * H' * (Tn - H * beta);   %beta的迭代公式

end


HTrain = RBFun(P,InputWeight,BiasofHiddenNeurons);

Y = HTrain * beta;
clear HTrain;


%%%%%% 计算训练准确度

TrainingAccuracy = sqrt(mse(T - Y))      %计算RMSE

