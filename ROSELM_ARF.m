clc;
clear;
%%%%%%��ʼ��������
NumberofHiddenNeurons = 100;    %������Ԫ��
N0 = 150;                        %���ڳ�ʼ���׶ε���������
Block = 10;                      %����ѧϰ ÿ����������
syms x;
%%%%%%����training set.
train_data = load('sinc_train');
T = train_data(:,1);                   %ѵ�����ĵ�14��Ϊ���
P = train_data(:,2);

clear train_data;

NumberofTrainingData = size(P,1);
NumberofInputNeurons = size(P,2);
%%%%����Ԥ������ռ�
H = zeros(1,1);
Y = zeros(1,1);
S = zeros(1,1);
gamma = zeros(1,1);

%%%%% ��ʼ���׶�
P0 = P(1:N0,:);           %ȡǰN0������
T0 = T(1:N0,:);
%%%%%%%�����������Ȩ��InputWeight(w_i)��ƫ�� BiasofHiddenNeurons(b_i)
InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons) * 2 - 1;

BiasofHiddenNeurons = rand(1,NumberofHiddenNeurons);    %��ͬ�����ƫ��ѡ��ͬ

H0 = RBFun(P0,InputWeight,BiasofHiddenNeurons);    %RBF�����
gamma0 = 1;                           %��ʼgamma0�������������
gamma(1,1:N0)=gamma0;
S0 = diag(gamma);

K = gamma0 .* H0' * H0 + N0 .* eye(NumberofHiddenNeurons);    %��ʽ12
beta0 = pinv(K) * H0' * S0 * T0;     %��ʽ12

store_H = H0;
store_Y = T0;
store_S = S0;
%%% FOLOO��ʼ��
% K_q = store_H' * store_H + eye(size(store_H,2)) .* N0;
% % beta_q = pinv(K_q) * store_H' * store_S * store_Y;
 %%%%%һ������ѧϰ
n= N0;
time=1; %ѭ������
Pn = P(n:(n+Block-1),:);
Tn = T(n:(n+Block-1),:);
H = RBFun(Pn,InputWeight,BiasofHiddenNeurons);

store_H = [store_H;H];
store_Y = [store_Y;Tn]; 

gamma_min = gamma0;%test��
K = K + H' * H * gamma_min + Block * eye(NumberofHiddenNeurons);%�����gamma��ô����
beta = beta0;
beta = beta + gamma_min * pinv(K) *H' *(Tn - H * beta) - Block * K * beta;
for i = 1:1:Block
gamma = [gamma,gamma_min];
end
store_S = diag(gamma);
%%%%һ��FOLOO

K_q = store_H' * store_H + eye(size(store_H,2)) .*( N0+Block*time);
beta_q = pinv(K_q) * store_H' * store_S * store_Y;

gamma_unknow=x;
K_a=K_q + H' * H * gamma_unknow + eye(NumberofHiddenNeurons)*Block;
G=diag(diag(gamma_unknow * H * pinv(K) * H'));
E=(eye(Block)-G)*(H * beta_q + gamma_unknow * H * pinv(K) * H' *(Tn - H * beta_q)-Block * H * pinv(K) * beta_q - Tn);
Objective_Function = 1/2 * E' * E;

%%%ţ�ٵ���

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

% %%%%%%%%%% ����ѧϰ�׶�
% for n = N0 : Block : NumberofTrainingData
%     if (n + Block - 1) > NumberofTrainingData
%         Pn = P(n:NumberofTrainingData,:);
%         Tn = T(NumberofTrainingData,:);
%         Block = size(Pn,1);
%         clear V;
%     else
%         Pn = P(n:(n+Block-1),:);
%         Tn = T(n:(n+Block-1),:);
%     end
%     H = RBFun(Pn,InputWeight,BiasofHiddenNeurons);
% %     %%%%%gamma
% %   gamma(1,N0+1:N0+Block) = Tempgamma;
%     
%     %%%%%����Kq,beta_q
%     TempK = H * H' + eye(NumberofHiddenNeurons) .* (N0 + n * Block); 
%     S_q = diag(gamma);
%     Y_q(N0+n,1) = Tn;
%     TempBeta =  pinv(TempK) * H' * S_q * Y_q;
%     %%%%%������ʽ
%     K = K + H' * H * gamma + m_k * eye(Block); %m_k�Ƕ���
%     beta = beta + gamma * pinv(K) * H' * (Y - H * beta) - m_k * pinv(K) * beta;   %beta�ĵ�����ʽ
% 
% end
% 
% 
% HTrain = RBFun(P,InputWeight,BiasofHiddenNeurons);
% 
% Y = HTrain * beta;
% clear HTrain;
% 
% 
% %%%%%% ����ѵ��׼ȷ��
% 
% TrainingAccuracy = sqrt(mse(T - Y))      %����RMSE

