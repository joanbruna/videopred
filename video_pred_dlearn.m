
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

Ltrain=round(3*size(X,2)/4);

[N,L]=size(X);


X1=X(:,1:end-2);
X2=X(:,2:end-1);
X3=X(:,3:end);

X0=[X1 ; X2; X3];
clear X1; 
clear X2;
clear X3;

X0=X0./repmat(sqrt(sum(X0.^2)), size(X0,1), 1) ;

II=randperm(size(X0,2));
X0 = X0(:,II);

param.lambda=0.05;
param.numThreads=12;
param.K=800;
param.batchsize=64;
param.iters=10000;
param.iter=10000;

D=mexTrainDL(X0(:,1:Ltrain),param);


Dbis=D(1:2*N,:);
Dpred=D(2*N+1:end,:);
Ybis=X0(1:2*N,Ltrain+1:end);
alphas = mexLasso(Ybis, Dbis, param);

Ypred = Dpred * alphas;
Ytarg = X0(2*N+1:end,Ltrain+1:end);

norm(Ypred(:)-Ytarg(:))/norm(Ytarg(:))

[errl,Al,predl]=linear_prediction(X,3:Ltrain,1);
X1 = X(:, Ltrain-1:end-2);
X2 = X(:,Ltrain:end-1);
X3 = X(:,Ltrain+1:end);
Ypredlin = Al * [X1;X2] ;



