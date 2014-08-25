
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

Ltrain=round(size(X,2)/2);

[N,L]=size(X);


X1=X(:,1:end-2);
X2=X(:,2:end-1);
Y=X(:,3:end);

X0=[X1 ; X2];
clear X1; 
clear X2;

%goal: predict Y from X
I=randperm(size(X,2)-2);
Xtrain=X0(:,I(1:Ltrain));
Ytrain=Y(:,I(1:Ltrain));

options.null=0;

tmp = load('~/matlab/video_prediction/dlearn_basic_dict.mat');
options.initdict = tmp.D;

[S, E, D, bias] = lista_regress(Xtrain, Ytrain, options);

keyboard;

Xtest=X0(:,I(Ltrain+1:end));
Ytest=Y(:,I(Ltrain+1:end));
Ypred = lista_predict(Xtest, S, E, D, bias, options);






