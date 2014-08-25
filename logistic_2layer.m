function [Theta, Pest, curve] = logistic_2layer( X0 , Y, options)
%this is a simple implementation of logsitic regression on multiple bits
% we estimate Y as logistic(Theta *X ) at first (later we will add a RELU layer).

[N, L]=size(X0);
mu = mean(X0,2);
X=zeros(N,L);
X=X0-repmat(mu,1,L);
%X(N+1,:)=ones(1,L);

[N, L]=size(X);
[NN, L]=size(Y);
batchsize = getoptions(options, 'batchsize', 32);
epochs = getoptions(options,'epochs',4);
lr = getoptions(options,'lr', 2e-4);
lambda = getoptions(options, 'lambda', 1e-2);

niters=L * epochs / batchsize;
verb = 100;

Nmid = 2*N;

%Theta{1} = randn(N);
%Theta{2} = randn(NN,N);
params{1} = eye(Nmid,N) + .1*randn(Nmid,N);
params{2} = randn(NN,Nmid);
params{3} = zeros(Nmid, 1);%bias
for l=1:size(params,2)
state.var{l} = 0*params{l};
state.gvar{l} = 0*params{l};
end

Y = (Y==1);
runCE1=0;
runCE1dumb = 0;
runCE2=0;
rast=1;
epsi = 1e-10;
probas = mean(Y,2);

for n=1:niters
init= mod( (n-1)*batchsize, L-batchsize+1); 
X0 = X(:,1+init:batchsize+init);
Y0= Y(:,1+init:batchsize+init);

%fprop
Z0 = params{1}*X0 + params{3}*ones(1,batchsize);
IZ0 = (Z0 > 0);
X1 = Z0.*IZ0;
Z1 = (1+ exp(-params{2}*X1)).^(-1);
%bprop
dparams{2} = -(Y0-Z1)*X1' + lambda * params{2};
dX1 = -(params{2}')*(Y0-Z1);
dparams{1} = (dX1.*IZ0)*X0' + 0*lambda * params{1};
dparams{3} = sum(dX1.*IZ0,2);

%update_params
[params, state]=update_params(params,dparams, lr, state);

Z1dumb = repmat(probas,1,batchsize);
Z1 = min(1-epsi,max(epsi, Z1));
runCE1 = runCE1 - sum( Y0(:).*log(Z1(:)) + (1-Y0(:)).*log(1-Z1(:))) ;
runCE1dumb = runCE1dumb - sum( Y0(:).*log(Z1dumb(:)) + (1-Y0(:)).*log(1-Z1dumb(:))) ;
runCE2 = runCE2 + (lambda/2)*(sum(params{2}(:).^2) + 0*sum(params{1}(:).^2));

if mod(n, verb)==verb-1
fprintf('iter %d of %d , error is %04f (%04f , %04f, %04f) \n', n+1, niters,runCE1+runCE2, runCE1, runCE1dumb, runCE2)
curve(rast)=runCE1;rast=rast+1;
%[std(Z1(:)) sum(IZ0(:))]

runCE1 = 0;
runCE1dumb = 0;
runCE2 = 0;
end

end

entropy = (1-probas).*log(1-probas)+(probas).*log(probas);
aventropy=-sum(entropy(:))*batchsize;

fprintf('Entropy lower bound is %f \n', aventropy)


Theta = params;
tmp = max(0,Theta{1}*X);
Pest = (1>exp(-Theta{2}*tmp));

end


