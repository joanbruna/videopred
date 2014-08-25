
close all;
clear all;

if 0
options.L=500000;
X= generate_jitter_data(options);
else
t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord32.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');
X=t1.Xout;%(:,1:1000000);
end

Ltrain=round(3*size(X,2)/4);

[N,L]=size(X);


options.null=0;
options.K=1600;
options.epochs=1;

[D, D0] = group_pooling_st(X, options);


