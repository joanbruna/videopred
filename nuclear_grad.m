function [dW,nn] = proximal_grad(Y, Z, X, W, k, alpha)
%this function computes the gradient descent of nuclear norm
%computed on the pooling space


%Y is size [P x bs]
%Z is size [M x bs]
%X is size [N x bs]
%W is size [M x N]

bs=size(Y,2);

%remove mean.
Y0 = Y - mean(Y,2)*ones(1,bs);

[U, S, V] = svd(Y0, 0);

dY = U * V';
dY = dY*(eye(bs) - bs^(-1)*ones(bs));
dY = dY + alpha * ones(size(Y));
dW = (Z .* replicate(dY./Y,k)) * X';
nn = sum(abs(diag(S)));



