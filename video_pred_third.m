
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');

%set a limit size
L=500000;
X=t1.Xout(:,1:L);


%linear prediction and compute prediction error
[X_l, Alinpred, Xres] = linear_prediction(X);

%whiten 
[X_w, mu_w, whiten, dewhiten ] = whiten_jojo(Xres);

%pooling transform
options.epochs=5;
options.proxloops=1;
options.k=2;
options.lr=1e-4;
options.wavelet_init = 1;
options.whitenmatrix = t1.whitenMatrix;
[X_p, X_pred, W, A, alpha] = pooling_layer(X_w, options);

%measure error
X_finlayer =  X_l + dewhiten*X_pred + repmat(mu_w, [1 L]) ;


fprintf('linear error is %f, pooling error is %f \n',norm(Xres(:))/norm(X(:)), norm(X_finlayer(:)- X(:))/norm(X(:)))













