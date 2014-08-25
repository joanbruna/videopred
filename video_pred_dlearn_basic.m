
close all;
clear all;

if 0
options.L=500000;
X= generate_jitter_data(options);
else
t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');
X=t1.Xout;%(:,1:1000000);
end

X=X./repmat(sqrt(sum(X.^2)), size(X,1), 1) ;

II=randperm(size(X,2));
X = X(:,II);

param.lambda=0.05;
param.numThreads=12;
param.K=300;
param.batchsize=64;
param.iters=10000;
param.iter=10000;

D=mexTrainDL(X,param);

