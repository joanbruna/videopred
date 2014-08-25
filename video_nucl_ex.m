
close all;
clear all;

%replace here by whatever data you want
t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');
X=t1.Xout;
%%%% X is the data matrix (each column is an example) %%%%

options.wavelet_init = 2;
options.whitenmatrix = t1.whitenMatrix;
options.dewhitenmatrix = t1.dewhitenMatrix;
options.epochs=2; %number of passes through the data
options.k=2; %groups (k=2 means 'complex' wavelets)
options.lr=4e-3;
options.reconstr=0;

[ W, W0, err_snap] = nuclear_learning(X, options);









