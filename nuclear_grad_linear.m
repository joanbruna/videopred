function [dW,nn] = nuclear_grad_linear(Z, X, W, alpha)
%this function computes the gradient descent of nuclear norm
%computed on the pooling space

%Z is size [M x bs]
%X is size [N x bs]
%W is size [M x N]

bs=size(Z,2);

%remove mean.
Z0 = Z - mean(Z,2)*ones(1,bs);

[U, S, V] = svd(Z0, 0);

dZ = U * V';
dZ = dZ*(eye(bs) - bs^(-1)*ones(bs));
dZ = dZ + alpha * sign(Z);
dW = dZ * X'; 
nn = sum(abs(diag(S)));



