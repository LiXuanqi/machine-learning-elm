clc;
clear;

train_data = load('sinc_train_gaussian_singularity.txt')
test_data = load('sinc_test.txt')

[TrainingTime, TrainingAccuracy] = elm_train('sinc_train_gaussian_singularity.txt', 0, 20, 'sig');
[TestingTime, TestingAccuracy] = elm_predict('sinc_test.txt');
load elm_output.mat

figure()
plot(test_data(:,2),output');
hold on
scatter(train_data(:,2),train_data(:,1),'r')
hold off
%%%%%SVM%%%%
%SVMSTRUCT = svmtrain(train_data(:,2), train_data(:,1));

%%%%%%%%%%BP%%%%%%
figure()

new_set=train_data(:,2)';
target_set=train_data(:,1)';


net=newff(minmax(new_set),[20,1],{'tansig','purelin'},'traingdm');
net.trainParam.time=6000000;
net.trainParam.epochs=1000; 
net.trainParam.show=500;
net.trainParam.min_grad=1e-10;
net.trainParam.mu=0.000001;
net.trainParam.lr=0.01;

[net,tr]=train(net,new_set,target_set);

O=sim(net,new_set);
%plot(new_set,target_set,'*',new_set,O);
RMSE = rms(O - target_set)

O_test = sim(net,test_data(:,2)');
plot(test_data(:,2)',test_data(:,1)','*',test_data(:,2)',O_test);

RMSE_test = rms(O_test - test_data(:,1)')
MAE_test = mae(test_data(:,1)' - O_test)
