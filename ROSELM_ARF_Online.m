clc;
clear;
%%%%%%初始参数设置
NumberofHiddenNeurons = 500;    %隐层神经元数
N0 = 494;                        %用于初始化阶段的样本数量
Block = 100;                      %在线学习 每的chunk所含数据量

%%%%%%载入training set.
load('Eng_H_0_Ma_0.mat')
%train_data = 0;
T = y_tr;                   %训练集的第1列为输出
P = x_tr;
NumberofTrainingData = size(P,1);

y=zeros(NumberofTrainingData,2);
y(2:NumberofTrainingData,1)=y_tr(1:NumberofTrainingData-1,1);
y(3:NumberofTrainingData,2)=y_tr(1:NumberofTrainingData-2,1);
P = [P,y];
P = P(3:NumberofTrainingData,:)
T = T(3:NumberofTrainingData,:)
NumberofTrainingData = size(P,1);


%%%%%%%数据归一化
% T = DataNormalized(T);
% P = DataNormalized(P);

clear train_data;


NumberofInputNeurons = size(P,2);
%%%%变量预先申请空间
% H = zeros(1,1);
% Y = zeros(1,1);
% S = zeros(1,1);
% gamma = zeros(1,1);

%%%%% 初始化阶段
P0 = P(1:N0,:);           %取前N0行数据
T0 = T(1:N0,:);
%%%%%%%随机生成输入权重InputWeight(w_i)和偏置 BiasofHiddenNeurons(b_i)
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons) * 2 - 1;  %产生[-1,1]的随机数

BiasofHiddenNeurons = rand(1,NumberofHiddenNeurons);    %不同激活函数偏置选择不同

H0 = RBFun(P0,InputWeight,BiasofHiddenNeurons);    %RBF激活函数
gamma0 = 2^10;                           
gamma(1,1:N0)=gamma0;
S0 = diag(gamma);

K0 = gamma0 .* H0' * H0 + N0 .* eye(NumberofHiddenNeurons);    %公式12
beta0 = eye(size(K0))/K0 * H0' * S0 * T0;     %公式12

%%%%%%%%存储前k-1次H,Y,S
store_H = H0;
store_Y = T0;
store_S = S0;


 %%%%%在线学习
 time = 0  
 gamma_min = gamma0;
 
 K = K0;
 beta = beta0;

 
 for n = N0+1 : Block : NumberofTrainingData
     time = time+1       %循环次数
%     if (n + Block - 1) > NumberofTrainingData
%         Pn = P(n:NumberofTrainingData,:);
%         Tn = T(NumberofTrainingData,:);
%         Block = size(Pn,1);
%        
%     else
        Pn = P(n:(n+Block-1),:);
        
        Tn = T(n:(n+Block-1),:);
%     end


H = RBFun(Pn,InputWeight,BiasofHiddenNeurons);

store_H = [store_H;H];
store_Y = [store_Y;Tn]; 

K = K + H' * H * gamma_min + Block * eye(NumberofHiddenNeurons);      %公式10

beta = beta + gamma_min * eye(size(K))/K *H' *(Tn - H * beta) - Block * eye(size(K))/K * beta;   %公式11
for i = 1:1:Block                       %存储每次的gamma值，形成S矩阵
gamma = [gamma,gamma_min];
end
store_S = diag(gamma);
%%%%一次FOLOO

K_q = store_H' * store_H + eye(size(store_H,2)) .*( N0+Block*time);   %公式14
beta_q = eye(size(K_q))/K_q * store_H' * store_S * store_Y;

gamma_unknow = [0.001 0.01 0.01 0.1 2^0 2^5 2^10 2^15 2^18];
for i = 1:length(gamma_unknow)
K_a=K_q + H' * H * gamma_unknow(i) + eye(NumberofHiddenNeurons)*Block;    %公式19
G=diag(diag(gamma_unknow(i) * H * eye(size(K_a))/K_a * H'));
E=eye(size(eye(Block)-G))/(eye(Block)-G)*(H * beta_q + gamma_unknow(i) * H * eye(size(K_a))/K_a * H' *(Tn - H * beta_q)-Block * H * eye(size(K_a))/K_a * beta_q - Tn);%公式23
Objective_Function(i) = 1/2 * E' * E;                                     %公式24
end

[minvalue,position]=min(Objective_Function)
gamma_min = gamma_unknow(position);

store_y = H * beta;

end

H_Train = RBFun(P,InputWeight,BiasofHiddenNeurons);


Y = H_Train * beta;

% 
% 
% %%%%%% 计算训练准确度
RD_train = abs((Y - T)./T);
Var_train=var(abs(Y-T))
Mean_train = mean(abs(Y-T))
STD_train = std(abs(Y-T))
%RD_test = (Y_test - T_test)./T_test;
% 
RMSE_train = sqrt(mse(T - Y))      %计算RMSE
%
figure(1)
plot(Y);
hold on
plot(y_tr)

%%%%%%%%test

T_test = y_te;
P_test= x_te;
NumberofTestData = size(P_test,1);
y_test=zeros(NumberofTestData,2);
P_test = [P_test,y_test];
for x = 1:1:NumberofTestData
    H_test = RBFun(P_test(x,:),InputWeight,BiasofHiddenNeurons);
    Y_once = H_test * beta;
    Y_test(x,:) = Y_once;
    P_test(x+1,8)=P_test(x,7);
    P_test(x+1,7)=Y_once;
end
% %%%%%% 计算测试准确度
RD_test = abs((Y_test - y_te)./y_te);
Var_test=var(abs(Y_test-y_te))
Mean_test = mean(abs(Y_test-y_te))
STD_test = std(abs(Y_test-y_te))
%RD_test = (Y_test - T_test)./T_test;
% 
RMSE_test = sqrt(mse(y_te - Y_test))      %计算RMSE
figure(2)
plot(Y_test);
hold on
plot(y_te)
hold off
