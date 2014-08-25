
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');



X=t1.Xout(:,1:400000);


[N, L] = size(X);

options.k=2;

I=get_legal_frames(X,options);

%compute covariance matrices
X1=0*X;
X2=0*X;
X1(:,2:end)=X(:,1:end-1);
X2(:,2:end)=X1(:,1:end-1);
S0=X*X';
S1=X*X1';
S2=X*X2';

B=[S1 S2];
U = [S0 S1' ; S1 S0];
Pred = B * pinv(U);

rien=[X1 ; X2];
Xestim= Pred * rien; 

X1=X(:,I);
X2=X(:,I+1);
M=400;
k=2;


%W=randn(M, N);
options.wavelet_init = 1;
options.whitenmatrix = t1.whitenMatrix;
options.dewhitenmatrix = t1.dewhitenMatrix;

[W] = init_pooling(X, M, k,options);
tol=1e-4;
Wd=pinv(W,tol);

[Z,ZZ]=pooling(X2,W,k);

niters=30;
epsi=1e-6;

%Y=X1;
Y=Xestim(:,I+1);
%epsi = 0.1;
%Y=X2 + epsi*randn(size(X2));
for n=1:niters
[Yp,YY] = pooling(Y, W, k); 
YY = YY .* replicate(Z./max(epsi,Yp), k);
Y = Wd * YY;
end





