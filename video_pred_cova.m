
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');

%set a limit size
L=100000;
X=t1.Xout(:,1:L);


%linear prediction and compute prediction error
[prederr] = linear_prediction(X, [3:L]);

X_cova = locallinearpred(X, 4);




