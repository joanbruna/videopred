function [ W, A, alpha, W0,Pred, err_snap] = pred_layer_nuclear(Xin, options,prior)

[N,L]=size(Xin);

M=getoptions(options,'M',2*N);
k=getoptions(options,'k',2);

%1- initialize the pooling operator
if nargin < 3
[W, A, alpha] = init_pooling(Xin, M, k,options);
else
W=prior.W;
A=prior.A;
alpha=prior.alpha;
end
W0 = W;

%2- stochastic gradient descent 
epochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',32);
batchnuclear=getoptions(options,'batchnuclear',8);
proxloops=getoptions(options,'proxloops',4);
lambda = getoptions(options,'lambda', 1);
lr = getoptions(options,'lr',1e-5);
lra = getoptions(options,'lra',1e-7);
gamma = getoptions(options,'lr_decay',0.98);

I=get_legal_frames(Xin,options);
 
niters = length(I)*epochs/batchsize;


curr_err =0;
rast=1;
alpha = -1e-3;

Wd=pinv(W,1e-3);

for n=1:niters
lr = lr * gamma^floor(n*batchsize/length(I));
init= mod( (n-1)*batchsize, length(I)-batchsize+1); 
I1=I(1+init:batchsize+init);
I2=I1+1;

X1=Xin(:,I1);
X2=Xin(:,I2);


%forward pass
[Z1,ZZ1]=pooling(X1,W,k);
[Z2,ZZ2]=pooling(X2,W,k);

if 0
%gradient descent predictor
V{1} = Z1 - Z2;
Q{1} = X1;
for l=1:proxloops
[Q{l+1},scratch{l}] = Ufp_mb(Q{l},V{l},W,alpha,k);
V{l+1} = pooling(Q{l+1},W,k) - Z2;
end
else
%alternating projections predictor
V{1} = Z2./Z1;
Q{1} = X1;
for l=1:proxloops
[Q{l+1},scratch{l}] = UUfp_mb(Q{l},V{l},W,Wd,k);
V{l+1} = Z2./ pooling(Q{l+1},W,k);
end

end
keyboard

if 0
%backward pass
curr_err = curr_err + norm(Q{proxloops+1}(:)-X2(:)).^2
G = Q{proxloops+1} - X2;
Pred(:,I2) = Q{proxloops+1};
dW=0*W;
dalpha=0;
for l=proxloops:-1:1
[G, dV0, dW0,dalpha0]=Ubp_mb(Q{l},V{l},W,alpha,G,k,scratch{l});
dW1 = pool_bp(dV0, W, Q{l}, scratch{l}.Y, scratch{l}.norms, k) - pool_bp(dV0, W, X2, ZZ2, Z2, k);
dW = dW + dW0 + dW1;
dalpha = dalpha + dalpha0;
end

init= round(rand*(L-batchnuclear));%mod( (n-1)*batchsize, length(I)-batchsize+1); 
I0 = 1+init:batchnuclear+init;
X0=Xin(:,I0);
[Y0,Z0]=pooling(X0,W,k);
[dW0, nn0] = nuclear_grad(Y0,Z0,X0,W,k, 0);


dW = dW + lambda * dW0;

%update parameters
W = W - lr * dW;
alpha = alpha - lra * dalpha;
end

%orthogonalize subspaces of W
%W = ortho_pools(W, k);

if mod(n,30)==29
fprintf('done %d of %d err %f \n', n, niters, curr_err)
err_snap(rast)=curr_err;rast=rast+1;
curr_err =0;
end

end





