
close all;
clear all;

options.L=500000;
if 0
X= generate_jitter_data(options);
else
t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');
X=t1.Xout(:,1:1000000);
end

[N,L]=size(X);

%dimension of subspaces
rho=getoptions(options,'rho',16);
M=getoptions(options,'M',N*rho);

I=3:L;

[errl,Al,predl]=linear_prediction(X,I,1);

fprintf('prediciton error in input space %f \n', errl)

W=randn(M,N);
Y=log((W*(X-predl)).^2);

%now do linear prediction on Y
[errq,Aq] = linear_prediction(Y, I,1);

%linear prediction error in the covariance space
fprintf('prediction error in covariance space %f \n', errq)

%try to reconstruct some samples in the original domain.
II=3+randperm(L-3);
Ibis = II(1:500);
chunk=X(:,Ibis);
chunk1=X(:,Ibis-1);
chunk2=X(:,Ibis-2);
Xpred_l = Al*[chunk1;chunk2];

qchunk1 = Y(:,Ibis-1);
qchunk2 = Y(:,Ibis-2);
Xpredq = Aq*[qchunk1;qchunk2];
Xpredq = exp(Xpredq);


tempo = 0*chunk;
tempo(:,2:end) = chunk(:,1:end-1) - Xpred_l(:,1:end-1);
fprintf('using legal initialization: \n')
[Xpr_q] = phase_recov(Xpredq, W, tempo, options);
fprintf('using illegal initialization: \n')
[Xpr_q] = phase_recov(Xpredq, W, chunk-Xpred_l, options);
norm(Xpr_q(:)-tempo(:))/norm(Xpr_q(:)-chunk(:)+Xpred_l(:))
%[Xpr_q] = phase_recov(Y(:,Ibis), W, chunk-Xpred_l, options);
%[Xpr_q] = phase_recov(qchunk, W, Xpred_l, options);

Xfulla = Xpr_q + Xpred_l;
Xfullb = -Xpr_q + Xpred_l;
mask = (sum((Xfulla-chunk).^2) < sum((Xfullb-chunk).^2));
mask = repmat(mask, size(Xfulla,1),1);
Xfull = Xfulla.*mask + Xfullb.*(1-mask);


% next steps: subsample in time : repredict 
% this should define a deeper and deeper architecture. 

%%another major question is how to cascade while keeping contraction

% role of sparsity? : perhaps replace the least squares linear predictor with sparse coding? how to do that?
%  




