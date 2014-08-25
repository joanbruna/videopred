function [Theta, Pest, curve] = logistic( X , Y, options)
%this is a simple implementation of logsitic regression on multiple bits
% we estimate Y as logistic(Theta *X ) at first (later we will add a RELU layer).


[N, L]=size(X);
[NN, L]=size(Y);
batchsize = getoptions(options, 'batchsize', 32);
epochs = getoptions(options,'epochs',4);
lr = getoptions(options,'lr', 5e-4);
lambda = getoptions(options, 'lambda', 1e-2);

niters=L * epochs / batchsize;
verb = 500;

Theta = randn(NN,N);
params{1} = Theta;
state.var{1} = 0*Theta;
state.gvar{1} = 0*Theta;

Y = (Y==1);
runCE1=0;
runCE2=0;
rast=1;
lrdecay = 0.9;

for n=1:niters
if mod(n, L/batchsize)==0
lr = lr * lrdecay ; 
end
init= mod( (n-1)*batchsize, L-batchsize+1); 
X0 = X(:,1+init:batchsize+init);
Y0= Y(:,1+init:batchsize+init);

%fprop
Z0 = (1+ exp(-params{1}*X0)).^(-1);
%bprop
dparams{1} = -(Y0-Z0)*X0' + lambda * params{1};

%update_params
[params, state]=update_params(params,dparams, lr, state);

runCE1 = runCE1 - sum( Y0(:).*log(Z0(:)) + (1-Y0(:)).*log(1-Z0(:))) ;
runCE2 = runCE2 + (lambda/2)*sum(params{1}(:).^2);

if mod(n, verb)==verb-1
fprintf('iter %d of %d , error is %04f (%04f , %04f) \n', n+1, niters,runCE1+runCE2, runCE1, runCE2)
curve(rast)=runCE1;rast=rast+1;
runCE1 = 0;
runCE2 = 0;
end

end

probas = mean(Y,2);
entropy = (1-probas).*log(1-probas)+(probas).*log(probas);
aventropy=-sum(entropy(:))*batchsize;

fprintf('Entropy lower bound is %f \n', aventropy)

Theta = params{1};
Pest = (1+exp(-Theta*X)).^(-1);

end


