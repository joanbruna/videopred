function [ W, A, alpha, W0,Pred, err_snap] = pred_layer(Xin, options,prior)

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
proxloops=getoptions(options,'proxloops',4);
lr = getoptions(options,'lr',1e-5);
lra = getoptions(options,'lra',1e-7);
gamma = getoptions(options,'lr_decay',0.98);

I=get_legal_frames(Xin,options);
 
niters = length(I)*epochs/batchsize;


curr_err =0;
curr_err0 =0;
rast=1;


for n=1:niters
%lr = lr * gamma^floor(n*batchsize/length(I));
init= mod( (n-1)*batchsize, length(I)-batchsize+1); 
I1=I(1+init:batchsize+init);
I0=I1-1;
I2=I1+1;

X1=Xin(:,I1);
X0=Xin(:,I0);
X2=Xin(:,I2);

%forward pass
[Z0,ZZ0]=pooling(X0,W,k);
[Z1,ZZ1]=pooling(X1,W,k);
V = A * (2*Z1 - Z0); %linear_prediction(Z0, Z1, A);
Q{1} = X1;
for l=1:proxloops
[Q{l+1},scratch{l}] = Ufp_mb(Q{l},V,W,alpha,k);
end

%backward pass
curr_err = curr_err + norm(Q{proxloops+1}(:)-X2(:)).^2;
%norm(X1(:)-X2(:))
curr_err0 = curr_err0 + norm(X1(:)-X2(:)).^2;
G = Q{proxloops+1} - X2;
Pred(:,I2) = Q{proxloops+1};
dV=0*V;
dW=0*W;
dalpha=0;
for l=proxloops:-1:1
[G, dV0, dW0,dalpha0]=Ubp_mb(Q{l},V,W,alpha,G,k,scratch{l});
dV = dV + dV0;
dW = dW + dW0;
dalpha = dalpha + dalpha0;
end

dA = dV  * (2*Z1' - Z0');
dV = A'*dV;
dW = dW + pool_bp(2*dV, W, X1, ZZ1, Z1, k) - pool_bp(dV, W, X0, ZZ0, Z0, k);

%update parameters
A = A - lr * dA; 
W = W - lr * dW;
alpha = alpha - lra * dalpha;

%orthogonalize subspaces of W
%W = ortho_pools(W, k);

if mod(n,30)==29
fprintf('done %d of %d err %f (naive %f) \n', n, niters, curr_err/(30*batchsize), curr_err0/(30*batchsize) )
err_snap(rast)=curr_err;rast=rast+1;
curr_err =0;
curr_err0 =0;
end

end





