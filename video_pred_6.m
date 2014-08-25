
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');


options.rho=16;

X=t1.Xout;%(:,1:100000);
I=get_legal_frames(X,options);

%return;

[N,L]=size(X);

%dimension of subspaces
rho=getoptions(options,'rho',20);
M=getoptions(options,'M',N*rho);

W=randn(M,N);

Y=(W*X).^2;


%now do linear prediction on Y

[errq,Aq] = linear_prediction(Y, I,1);

%linear prediction error in the covariance space
fprintf('prediction error in covariance space %f \n', errq)

%quadratic_pred(X, options);
[errl,Al]=linear_prediction(X,I,1);

fprintf('prediciton error in input space %f \n', errl)

%try to reconstruct some samples in the original domain.
Ibis = I(1:400);
chunk=X(:,Ibis);
chunk1=X(:,Ibis-1);
chunk2=X(:,Ibis-2);
Xpred_l = Al*[chunk1;chunk2];

qchunk=(W*chunk).^2;
qchunk1=(W*chunk1).^2;
qchunk2=(W*chunk2).^2;
Xpredq = Aq*[qchunk1;qchunk2];

[Xpr_q] = phase_recov(Xpredq, W, Xpred_l, options);
%[Xpr_q] = phase_recov(qchunk, W, Xpred_l, options);

rien = zeros(N,4*400); 
rien(:,1:4:end)=chunk;
rien(:,2:4:end)=Xpred_l;
rien(:,3:4:end)=Xpr_q;
rien(:,4:4:end)=chunk1;

totjunt = t1.dewhitenMatrix * rien ;






