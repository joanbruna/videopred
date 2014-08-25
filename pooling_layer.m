function [Xpool, Xpred,W2 , A2, alpha2 ]= pooling_layer(X, options)

[W, A, alpha] = pred_layer(X, options);

if 1
%add another prox step
options.lr=3e-5;
options.proxloops=2;
prior.W=W;
prior.A=A;
prior.alpha=alpha/2;
[W2, A2, alpha2,~,Xpred, erro2] = pred_layer(X, options, prior);
end

Xpool = pooling(X, W2, options.k);



