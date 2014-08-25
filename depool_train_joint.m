
close all;
clear all;

load('/misc/vlgscratch3/LecunGroup/bruna/vocpatches.mat');

[N, L] = size(X);


X= X - repmat(mean(X), size(X,1), 1);

K=2*N;
W=randn(K,N);

Y=(W*X);

Ltrain = 500000;

Ytrain = Y(:,1:Ltrain);
Ytest = Y(:,Ltrain+1:end);

P = sign(Ytrain);
M = abs(Ytrain);
Pt = sign(Ytest);
Mt = abs(Ytest);

mu = mean(sqrt(sum(M.^2)));
M = M/mu; 
Mt = Mt/mu;


%rather than predicting P directly, we are going to predict phase differences. 
%with enough phase differences, we can recover an estimate of P. Phase differences 
%do not depend upon the arbritrary global sign

rho = 2; %redundancy factor for phase differences.
PD = zeros(K*rho,size(Ytrain,2));
shi=[1 K/2];
SAT = zeros(rho*K, K);
for r=1:rho
PD(1+(r-1)*K:r*K,:) = P .* circshift(P, [shi(r) 0]);
SAT(1+(r-1)*K:r*K,:) = eye(K) - circshift(eye(K), [0 shi(r)]) ;
end

%keyboard;

options.epochs=8;

[Theta, Pest, curva] = logistic( M, PD, options);

Pred = greedy_phase_reduction(Pest, SAT);







