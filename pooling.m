function [Z, Z0] = pooling(X, W, k)

[N, bs]=size(X);
[M, N] =size(W);
P=M/k;

Z0 = W*X ; 
tmp = reshape(Z0,k,P*bs);
Z = reshape(sqrt(sum(tmp.^2)),P,bs);




