
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');


%%% learning spatio-temporal pooling
if 0 
options.K=300;
options.initdictionary=t2.Dout;
options.epochs=2;
nexs = 60000;

X = prune_input(t1.Xout, nexs);

D = group_pooling_st(X, options);
end
%%%%%%%%%%%



%%%% best linear prediction

%identify transitions
O=1000;
Xtmp = 0*t1.Xout;
Xtmp(:,2:end)=t1.Xout(:,1:end-1);
Xtmp = t1.Xout - Xtmp;
normes=sqrt(sum(Xtmp.^2));
normes1=0*normes;
normes2=0*normes;
normes1(2:end)=normes(1:end-1);
normes2(1:end-1)=normes(2:end);
normbis = normes./(normes1+normes2);
[val,pos]=sort(normbis,'descend');
outliers=pos(1:O);


%compute best linear prediction 

%compute covariance matrices
[N,L]=size(t1.Xout);
X1=0*t1.Xout;
X2=0*t1.Xout;
X1(:,2:end)=t1.Xout(:,1:end-1);
X2(:,2:end)=X1(:,1:end-1);
S0=t1.Xout*t1.Xout';
S1=t1.Xout*X1';
S2=t1.Xout*X2';

B=[S1 S2];
U = [S0 S1' ; S1 S0];
Pred = B * pinv(U);

rien=[X1 ; X2];
Xestim= Pred * rien; 

%%%%%compute best linear prediction of the form hat(x_t) = A (2x_t-1 - x_t-2)
Y = 2*X1 - X2;
S1bis = t1.Xout*Y';
S0bis = Y*Y';
A = S1bis * pinv(S0bis);
Xestim_base = A * Y; 

%compare the two prediction errors
norm(t1.Xout(:)-Xestim(:))
norm(t1.Xout(:)-Xestim_base(:))









