function [S, E, D, bias] = lista_regress(X, Y, options)

[N, L]=size(X);
[NN, L]=size(Y);
batchsize = getoptions(options, 'batchsize', 128);
lambda = getoptions(options, 'lambda', 0.02);
steps = getoptions(options, 'steps',1);
epochs = getoptions(options,'epochs',3);
lr = getoptions(options,'lr', 1e-5);

niters=L * epochs / batchsize;
verb = 100;

if isfield(options,'initdict')
E=[[options.initdict' 0*options.initdict'];[0*options.initdict' options.initdict']]; 
emax = max(svd(E))^2;
E = .5 * E / emax ; 
S = eye(size(E,1)) - emax * E*E';
D = [options.initdict options.initdict] / emax;
K= size(E,1);
bias = 0*rand(K,1);


else
%init
K = getoptions(options,'K', 4*N);
E=randn(K,N);
E = .1 * E / max(svd(E));
S=eye(K); % + .1*randn(K);
D=randn(NN,K);
bias = 0*rand(K,1);
end




state.var{1} = 0*S;
state.gvar{1} = 0*S;
state.var{2} = 0*E;
state.gvar{2} = 0*E;
state.var{3} = 0*bias;
state.gvar{3} = 0*bias;
state.var{4} = 0*D;
state.gvar{4} = 0*D;

runG=0;
runY=0;

for n=1:niters
init= mod( (n-1)*batchsize, L-batchsize+1); 
X0 = X(:,1+init:batchsize+init);
Y0= Y(:,1+init:batchsize+init);

bibi = repmat(bias,1,batchsize);

%fprop
B= E * X0;
Z{1} = (abs(B) > bibi).*(B - sign(B).*bibi);
for s=1:steps
[C{s+1}, Z{s+1} ] = lista_fp(Z{s}, S, B, bibi);
end
U = D * Z{steps+1};

%bprop
G = U - Y0;
dS=0;
dB=0;
dD=0;
dbias=0*bibi;
dD = G * Z{steps+1}';
dZ = D' * G + lambda * sign(Z{steps+1});
for s=steps:-1:1
[dZ, dS, dB, dbias] = lista_bp(C{s+1}, Z{s}, S, B, bibi, dZ, dS, dB, dbias); 
end
dB = dB + (abs(B) > bibi).* dZ ;
dbias = dbias - sign(B).*(abs(B) > bibi).*dZ ; 
dE = dB * X0';

%update_params
[S, E, bias, D, state]=update_params(S, E, bias, D, dS, dE, sum(dbias,2), dD, lr, state);

runG = runG + norm(G(:));
runY = runY + norm(Y0(:));

if mod(n, verb)==verb-1
fprintf('iter %d of %d , error is %f \n', n+1, niters,runG/runY)
runG = 0;
runY = 0;
fprintf('sparsity adjust %f \n', sum((abs(Z{steps+1}(:))>0))/numel(Z{steps+1}(:)))
%fprintf('mean bias is %f [%f %f] \n', mean(bias(:)), min(bias(:)), max(bias(:)))
end

end

end



function [C, Z] = lista_fp ( Zin, S, B, bibi)

C = B + S * Zin;
Z = (abs(C) > bibi).*(C - sign(C).*bibi);

end

function [dZ, dS, dB, dbias] = lista_bp(Cin, Zin, S, B, bibi, dZin, dSin, dBin, dbiasin)

dC = (abs(Cin) > bibi).*dZin;
dB = dBin + dC;
dbias = dbiasin - sign(Cin).*dC;
dS = dSin + dC * Zin';
dZ = S' * dC;

end

function [S, E, bias, D, stateout] = update_params(Sin, Ein, biasin, Din, dS, dE, dbias, dD, lr, statein)

stateout=statein;
rho = 0.9;
epsilon = 1e-6;
if 1
%vanilla SGD
S = Sin - lr * dS;
E = Ein - lr * dE;
bias = biasin - lr * dbias ;
D = Din - lr * dD;

elseif 1

stateout.var{1} = rho * stateout.var{1} - lr * dS;
stateout.var{2} = rho * stateout.var{2} - lr * dE;
stateout.var{3} = rho * stateout.var{3} - lr * dbias;
stateout.var{4} = rho * stateout.var{4} - lr * dD;

S=Sin+stateout.var{1};
E=Ein+stateout.var{2};
bias=biasin+stateout.var{3};
D=Din+stateout.var{4};


else
%ADADELTA
%(S, E, bias, D)

dS2 = dS.^2;
dE2 = dE.^2;
dbias2 = dbias.^2;
dD2 = dD.^2;
stateout.gvar{1}= rho * stateout.gvar{1} + (1-rho) * dS2;
stateout.gvar{2}= rho * stateout.gvar{2} + (1-rho) * dE2;
stateout.gvar{3}= rho * stateout.gvar{3} + (1-rho) * dbias2;
stateout.gvar{4}= rho * stateout.gvar{4} + (1-rho) * dD2;
deltaS = - (sqrt(epsilon+stateout.var{1}) ./ sqrt(epsilon+dS2)) .* dS;
deltaE = - (sqrt(epsilon+stateout.var{2}) ./ sqrt(epsilon+dE2)) .* dE;
deltabias = - (sqrt(epsilon+stateout.var{3}) ./ sqrt(epsilon+dbias2)).* dbias;
deltaD = - (sqrt(epsilon+stateout.var{4}) ./ sqrt(epsilon+dD2)) .* dD;
stateout.var{1} = rho * stateout.var{1} + (1-rho) * (deltaS.^2);
stateout.var{2} = rho * stateout.var{2} + (1-rho) * (deltaE.^2);
stateout.var{3} = rho * stateout.var{3} + (1-rho) * (deltabias.^2);
stateout.var{4} = rho * stateout.var{4} + (1-rho) * (deltaD.^2);
S = Sin + deltaS;
E = Ein + deltaE;
bias = biasin + deltabias;
D = Din + deltaD;

end

D=D./repmat(sqrt(sum(D.^2)), size(D,1),1);

end



