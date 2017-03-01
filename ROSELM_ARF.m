clc;
clear;
%%%%%%初始参数设置
NumberofHiddenNeurons = 20;    %隐层神经元数
N0 = 3000;                        %用于初始化阶段的样本数量
Block = 100;                      %在线学习 每步的数据量
syms x;
%%%%%%载入training set.
train_data = load('sinc_train');
T = train_data(:,1);                   %训练集的第14列为输出
P = train_data(:,2);

clear train_data;

NumberofTrainingData = size(P,1);
NumberofInputNeurons = size(P,2);
%%%%变量预先申请空间
H = zeros(1,1);
Y = zeros(1,1);
S = zeros(1,1);
gamma = zeros(1,1);

%%%%% 初始化阶段
P0 = P(1:N0,:);           %取前N0行数据
T0 = T(1:N0,:);
%%%%%%%随机生成输入权重InputWeight(w_i)和偏置 BiasofHiddenNeurons(b_i)
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons) * 2 - 1;

BiasofHiddenNeurons = rand(1,NumberofHiddenNeurons);    %不同激活函数偏置选择不同

H0 = RBFun(P0,InputWeight,BiasofHiddenNeurons);    %RBF激活函数
gamma0 = 1;                           %初始gamma0可以任意给定吗
gamma(1,1:N0)=gamma0;
S0 = diag(gamma);

K = gamma0 .* H0' * H0 + N0 .* eye(NumberofHiddenNeurons);    %公式12
beta0 = pinv(K) * H0' * S0 * T0;     %公式12

store_H = H0;
store_Y = T0;
store_S = S0;
%%% FOLOO初始化
% K_q = store_H' * store_H + eye(size(store_H,2)) .* N0;
% % beta_q = pinv(K_q) * store_H' * store_S * store_Y;
 %%%%%一次在线学习
 time = 0
 gamma_min = gamma0;
 beta = beta0;
 gamma_unknow=x;
 for n = N0 : Block : NumberofTrainingData
     time = time+1
    if (n + Block - 1) > NumberofTrainingData
        Pn = P(n:NumberofTrainingData,:);
        Tn = T(NumberofTrainingData,:);
        Block = size(Pn,1);
        clear V;
    else
        Pn = P(n:(n+Block-1),:);
        Tn = T(n:(n+Block-1),:);
    end
Pn = P(n:(n+Block-1),:);
Tn = T(n:(n+Block-1),:);
H = RBFun(Pn,InputWeight,BiasofHiddenNeurons);

store_H = [store_H;H];
store_Y = [store_Y;Tn]; 

K = K + H' * H * gamma_min + Block * eye(NumberofHiddenNeurons);%这里的gamma怎么处理

beta = beta + gamma_min * pinv(K) *H' *(Tn - H * beta) - Block * K * beta;
for i = 1:1:Block
gamma = [gamma,gamma_min];
end
store_S = diag(gamma);
%%%%一次FOLOO

K_q = store_H' * store_H + eye(size(store_H,2)) .*( N0+Block*time);
beta_q = pinv(K_q) * store_H' * store_S * store_Y;


K_a=K_q + H' * H * gamma_unknow + eye(NumberofHiddenNeurons)*Block;
G=diag(diag(gamma_unknow * H * pinv(K) * H'));
E=(eye(Block)-G)*(H * beta_q + gamma_unknow * H * pinv(K) * H' *(Tn - H * beta_q)-Block * H * pinv(K) * beta_q - Tn);
Objective_Function = 1/2 * E' * E;

%%%牛顿迭代

x1=10;
eps = 0.01;
f=Objective_Function ;
grad1=jacobian(f,x); 
grad2=jacobian(grad1,x); 
k=0; 
while 1    
    grad1z=subs(subs(grad1,x,x1)); 
   grad2z=subs(subs(grad2,x,x1));
    x2=x1-inv(grad2z)*(grad1z)'; 
    if norm(x1-x2)<eps     
        break; 
    else
       
        k=k+1;  
        x1=x2; 

        
    end
end
gamma_min = x1;
 end

HTrain = RBFun(P,InputWeight,BiasofHiddenNeurons);
% 
Y = HTrain * beta;

% 
% 
% %%%%%% 计算训练准确度
% 
TrainingAccuracy = sqrt(mse(T - Y))      %计算RMSE

