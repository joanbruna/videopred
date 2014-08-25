function [Z, Zp, Z0] = poolingphase(X, W, k)

[N, bs]=size(X);
[M, N] =size(W);
P=M/k;

Z0 = W*X ; 
tmp = reshape(Z0,k,P*bs);
Z = reshape(sqrt(sum(tmp.^2)),P,bs);
Zp = Z0./replicate(Z,k);



