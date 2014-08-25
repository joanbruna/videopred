function [ W, W0, err_snap] = nuclear_learning(Xin, options,prior)

[N,L]=size(Xin);

M=getoptions(options,'M',2*N);
k=getoptions(options,'k',2);

%1- initialize the pooling operator
if nargin < 3
[W, A, alpha] = init_pooling(Xin, M, k,options);
else
W=prior.W;
end
W0 = W;
Wd = pinv(W);

%2- stochastic gradient descent 
epochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',8);
batchsize2=getoptions(options,'batchsize',16);
proxloops=getoptions(options,'proxloops',4);
lr = getoptions(options,'lr',1e-5);
gamma = getoptions(options,'lr_decay',0.98);
MM = getoptions(options,'maxnn',batchsize);
beta = getoptions(options,'beta',0.75);
kappa = getoptions(options,'kappa',2e-1);

%I=get_legal_frames(Xin,options);
I=randperm(L); 

reconstr=getoptions(options,'reconstr',0);

niters = length(I)*epochs/batchsize;


curr_err =0;
count = 0;
rast=1;

%niters=50000;

for n=1:niters
if mod(n, round(length(I)/batchsize))==0
lr = lr * gamma;
end
init= round(rand*(L-batchsize));%mod( (n-1)*batchsize, length(I)-batchsize+1); 
I0 = 1+init:batchsize+init;
init= round(rand*(L-batchsize2));%mod( (n-1)*batchsize, length(I)-batchsize+1); 
I1=I(1+init:batchsize2+init);
X0=Xin(:,I0);
X1=Xin(:,I1);

if reconstr

[Y0,Z0]=pooling(X0,W,k);
[Y1,Z1]=pooling(X1,W,k);
[dW0, nn0] = nuclear_grad(Y0,Z0,X0,W,k, 0.01);
[dW1, nn1] = nuclear_grad(Y1,Z1,X1,W,k, 0.01);
ss=svd(X1,0);MM=1.0*sum(abs((ss)));
ss=svd(X0,0);MM0=1.0*sum(abs((ss)));lambda = 3*MM0/MM;
curr_err = curr_err + nn0 + max(0, MM - nn1);

%simplest possible architecture
dW = dW0 - lambda * dW1.*(nn1 <= MM);
G = ( Wd * Z1 - X1);
Gb = G*X1';
dW2 = Wd'*Gb;
dWd = Gb*W';
kappa = .5 * norm(dW(:))/norm(dW2(:)) ;
W = W - lr * (dW + kappa * dW2);
Wd = Wd - lr * dWd;
curr_err = curr_err + kappa*norm(G(:));

Wd=Wd./repmat(sqrt(sum(Wd.^2,1)),[size(Wd,1) 1]);

else
%forward pass
[Y0,Z0]=pooling(X0,W,k);
[Y1,Z1]=pooling(X1,W,k);

[dW0, nn0] = nuclear_grad(Y0,Z0,X0,W,k, 0.01);
[dW1, nn1] = nuclear_grad(Y1,Z1,X1,W,k, 0.01);
ss=svd(X1,0);MM=1.0*sum(abs((ss)));
ss=svd(X0,0);MM0=1.0*sum(abs((ss)));lambda = 3*MM0/MM;
curr_err = curr_err + nn0 + lambda * max(0, MM - nn1);

%simplest possible architecture
dW = dW0 - lambda * dW1.*(nn1 <= MM);
W = W - lr * dW;
count = count + (nn1 <= MM);

if 1
W=W./repmat(sqrt(sum(W.^2,2)),[1 size(W,2)]);
else
[uu,ss,vv]=svd(W,0);
W=uu*(ss.^beta)*vv';
end

end

%orthogonalize subspaces of W
%W = ortho_pools(W, k);

if mod(n,500)==499
fprintf('done %d of %d err %f norm W %f count %d \n', n, niters, curr_err, norm(W(:)), count)
err_snap(rast)=curr_err;rast=rast+1;
curr_err =0;count=0;
end

end





