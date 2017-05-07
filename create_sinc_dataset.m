clear;
clc;
%%%%%%%%%%%%%%%%%%test_sinc%%%%%%%%%%%%%%%%%%%
x=linspace(-4,4,150)
sinc=sin(pi*x)./(pi*x);
%plot(x,sinc,'b')
test_data = [sinc',x'];
save sinc_test.txt test_data -ascii

clear;
clc;
%%%%%train_sinc%%%%%%%%%%%%%%%%

x=linspace(-4,4,200)
sinc=sin(pi*x)./(pi*x);
sinc = sinc'

% figure()
% scatter(x,sinc,'k')
% hold on
sinc = sinc + mvnrnd (0,0.01,200);  %�����ֵΪ0������0.01�ĸ�˹�ֲ��������ź�
% scatter(x,sinc,'r')
% hold on
train_test = [sinc,x'];
save sinc_train_gaussian.txt train_test -ascii
%%%%����(-1,1)�������
A=rand(1,20)'
B=-1+2.*A  %�������䵽��-1��1��
sinc(1:10:200,1) = sinc(1:10:200,1) + B;
train_singularity = [sinc,x']
singularity = [sinc(1:10:200,1),x(1,1:10:200)']
% scatter(x(1,1:10:200),sinc(1:10:200,1),'g')
% hold off

scatter(x,sinc,'r')
save sinc_train_gaussian_singularity.txt train_singularity -ascii
save sinc_singularity.txt singularity -ascii