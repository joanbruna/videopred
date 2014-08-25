
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');



X=t1.Xout(:,1:400000);

options.wavelet_init = 1;
options.whitenmatrix = t1.whitenMatrix;
options.dewhitenmatrix = t1.dewhitenMatrix;
options.epochs=2;
options.proxloops=3;
options.k=2;
options.lr=1e-4;

[W, A, alpha, W0, P, erro] = pred_layer_nuclear(X, options);

if 0
%add another prox step
options.lr=3e-5;
options.proxloops=2;
prior.W=W;
prior.A=A;
prior.alpha=alpha/2;
[W2, A2, alpha2,~,P2, erro2] = pred_layer(X, options, prior);
end


X2 = pooling(X, W2, options.k);


if 0
%%%%%compute best linear prediction of the form hat(x_t) = A (2x_t-1 - x_t-2)
X1=0*X;
X2=0*X;
X1(:,2:end)=X(:,1:end-1);
X2(:,2:end)=X1(:,1:end-1);
Y = 2*X1 - X2;
S1bis = X*Y';
S0bis = Y*Y';
A = S1bis * pinv(S0bis);
Xestim_base = A * Y; 
end








