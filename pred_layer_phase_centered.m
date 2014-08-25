function [ W, Am, Ap, Wd,err_snap] = pred_layer_phase_centered(Xin, options,prior)

[N,L]=size(Xin);

k=getoptions(options,'k',2);
M=getoptions(options,'M',k*N);
P=M/k;

%1- initialize the pooling operator
if nargin < 3
[W] = init_pooling(Xin, M, k,options);
%Am0 = eye(P); 
%Am1 = eye(P); 
%Ap0 =eye(M) ;  
%Ap1 =eye(M) ;  
Am = eye(P); 
Ap =eye(M) ;  
Wd = pinv(W);
else
W=prior.W;
Am=prior.Am;
Ap=prior.Ap;
%Am0=prior.Am0;
%Ap0=prior.Ap0;
%Am1=prior.Am1;
%Ap1=prior.Ap1;
Wd=prior.Wd;
end
W0 = W;

%2- stochastic gradient descent 
epochs=getoptions(options,'epochs',4);
batchsize=getoptions(options,'batchsize',32);
lr = getoptions(options,'lr',1e-5);
gamma = getoptions(options,'lr_decay',0.98);
alpha = getoptions(options,'alpha',0.1);

I=get_legal_frames(Xin,options);
niters = length(I)*epochs/batchsize;

curr_err =0;
curr_err_n =0;
rast=1;

for n=1:niters
if mod(n, round(length(I)/batchsize))==0
lr = lr * gamma;
end

init= mod( (n-1)*batchsize, length(I)-batchsize+1); 
I1=I(1+init:batchsize+init);
I2=I1+1;
I0=I1-1;

X0=Xin(:,I0);
X1=Xin(:,I1);
X2=Xin(:,I2);

%centered estimation this time

%forward pass
[Z0,Z0p, ZZ0]=poolingphase(X0,W,k);
[Z2,Z2p, ZZ2]=poolingphase(X2,W,k);

%Um = Am0 * Z0 + Am1 * Z1;
%Up0 = Ap0 * Z0p + Ap1 * Z1p;
Um = +Am * Z0 + 1*Am * Z2;
Up0 = +Ap * Z0p + 1*Ap * Z2p;
Up_m = reshape(sqrt(sum(reshape(Up0,k,P*batchsize).^2)),P,batchsize);
Up = Up0 ./ replicate(Up_m,k);

Umm = replicate(Um, k);
R = Umm .* Up;
Q = Wd * R;

%backward pass
curr_err = curr_err + norm(Q(:)-X1(:)).^2;
curr_err_n = curr_err_n + norm(X1(:)).^2;
G = Q - X1;

%G is [N x bs]
%Z0 is [P x bs]
%Z0p is [M x bs]
%Um  is [P x bs]
%Up is [M x bs]
%R is [M x bs]

dWd = G * R'; 
dR = Wd' * G;
dUp = Umm .* dR;
dUmm = Up .* dR;
dUp0 = phasebp(dUp, Up, Up_m, k);

dUm = reshape(sum(reshape(dUmm,k,batchsize*P)),P, batchsize);
dZ0p = +Ap' * dUp0;
dZ2p = 1*Ap' * dUp0;
%dAp0 = dUp0 * Z0p';
%dAp1 = dUp0 * Z1p';
dAp = 1*dUp0 * Z2p' + dUp0 * Z0p';
dZ0 = +Am' * dUm;
dZ2 = 1*Am' * dUm;
%dAm0 = dUm * Z0';
%dAm1 = dUm * Z1';
dAm = 1*dUm * Z2' + dUm * Z0';

%add some sparsity on modules
dZ0 = dZ0 + alpha ;
dZ2 = dZ2 + alpha ;

dW0 = poolingphase_bp(Z0,Z0p, ZZ0, dZ0, dZ0p, X0, W, k);
dW2 = poolingphase_bp(Z2,Z2p, ZZ2, dZ2, dZ2p, X0, W, k);
dW = dW0 + dW2;

%update parameters
W = W - lr * dW;
Wd = Wd - lr * dWd;
Am = Am - lr * dAm;
Ap = Ap - lr * dAp;
%Am0 = Am0 - lr * dAm0;
%Am1 = Am1 - lr * dAm1;
%Ap0 = Ap0 - lr * dAp0;
%Ap1 = Ap1 - lr * dAp1;

if mod(n,500)==499
fprintf('done %d of %d err %f (lr=%f) \n', n+1, niters, curr_err/curr_err_n, lr)
err_snap(rast)=curr_err/curr_err_n;rast=rast+1;
curr_err =0;
curr_err_n =0;
end

end

%orthogonalize subspaces of W
%W = ortho_pools(W, k);


end





