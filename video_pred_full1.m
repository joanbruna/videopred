
close all;
clear all;

Ltrain=1000000;
Ltest=1000000;
options.null=0;

if 0
options.L=Ltrain;
Xtrain= generate_jitter_data(options);
options.L=Ltest;
Xtest= generate_jitter_data(options);
else
t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');
X=t1.Xout(:,1:500000);
end

[N,L]=size(X);

%dimension of subspaces
rho=getoptions(options,'rho',16);
M=getoptions(options,'M',N*rho);

I=3:L;

[errl,Al,predl]=linear_prediction(X,I,1);

fprintf('prediciton error in input space %f \n', errl)

W=randn(M,N);
Y=(W*(X-predl)).^2;

%now do linear prediction on Y
[errq,Aq] = linear_prediction(Y, I,1);

%linear prediction error in the covariance space
fprintf('prediction error in covariance space %f \n', errq)

%try to reconstruct some samples in the original domain.
Ibis = I(1:500);
chunk=X(:,Ibis);
chunk1=X(:,Ibis-1);
chunk2=X(:,Ibis-2);
Xpred_l = Al*[chunk1;chunk2];

%chunk1=errl(:,Ibis-1);
%chunk2=X(:,Ibis-2);
%qchunk=(W*chunk).^2;
%qchunk1=(W*chunk1).^2;
%qchunk2=(W*chunk2).^2;
qchunk1 = Y(:,Ibis-1);
qchunk2 = Y(:,Ibis-2);
Xpredq = Aq*[qchunk1;qchunk2];

[Xpr_q] = phase_recov(Xpredq, W, randn(size(chunk-Xpred_l)), options);
%[Xpr_q] = phase_recov(Y(:,Ibis), W, chunk-Xpred_l, options);
%[Xpr_q] = phase_recov(qchunk, W, Xpred_l, options);

Xfulla = Xpr_q + Xpred_l;
Xfullb = -Xpr_q + Xpred_l;
mask = (sum((Xfulla-chunk).^2) < sum((Xfullb-chunk).^2));
mask = repmat(mask, size(Xfulla,1),1);
Xfull = Xfulla.*mask + Xfullb.*(1-mask);

if 0
rien = zeros(N,4*400); 
rien(:,1:4:end)=chunk;
rien(:,2:4:end)=Xpred_l;
rien(:,3:4:end)=Xpr_q;
rien(:,4:4:end)=chunk1;
end


% next steps: subsample in time : repredict 
% this should define a deeper and deeper architecture. 

%%another major question is how to cascade while keeping contraction

% role of sparsity? : perhaps replace the least squares linear predictor with sparse coding? how to do that?
%  




