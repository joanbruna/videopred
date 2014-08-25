function dW = poolingphase_bp(Z,Zp, ZZ, dZ, dZp, X, W, k)
%dW0 = poolingphase_bp(Z0,Z0p, ZZ0, dZ0, dZ0p, X0, W, k);

%ZZ is [M x bs]
%Z is [P x bs]
%Zp is [M x bs]
%dZ is [P x bs]
%dZp is [M x bs]

[P, bs] = size(dZ);
[M, bs] = size(Zp);

dWm = (Zp .* replicate(dZ,k) ) * X';
dY = (dZp - Zp .* reshape(repmat(sum(reshape( dZp.*Zp, k, P*bs)),[k 1]),M, bs))./replicate(Z,k);
dW = dWm + dY * X' ; 


