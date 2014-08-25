
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');
X=t1.Xout;%(:,1:400000);

%options.N=128;
%options.L=400000;
%options.M=128;

%[X, dewhiten] = generate_jitter_data(options);

options.epochs=1;
options.wavelet_init = 2;
options.lr=4e-3;

Xtiny = X(:,1:500000);
[ Wl ] = nuclear_learning_linear(Xtiny, options);

tol = 1e-1;
project = pinv(Wl, tol) * Wl;
X = X - project * X;

keyboard;

options.whitenmatrix = t1.whitenMatrix;
options.dewhitenmatrix = t1.dewhitenMatrix;
options.k=2;
%options.M=480;
%options.lr=6e-4;
%options.kappa=2e-1;
options.reconstr=0;

[ W, W0, err_snap] = nuclear_learning(X, options);

aviam = W * options.dewhitenmatrix' ; 

displayPatches(aviam');







