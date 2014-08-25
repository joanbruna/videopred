
close all;
clear all;

load('/misc/vlgscratch3/LecunGroup/bruna/vocpatches.mat');

[N, L] = size(X);

K=2*N;
W=randn(K,N);


Y=(W*X);

Ltrain1 = 300000;
Ltrain2 = 600000;

Ytrain1 = Y(:,1:Ltrain1);
Ytrain2 = Y(:,Ltrain1+1:Ltrain2);
Ytest = Y(:,Ltrain2+1:end);

P1 = sign(Ytrain1);
M1 = abs(Ytrain1);
P2 = sign(Ytrain2);
M2 = abs(Ytrain2);
Pt = sign(Ytest);
Mt = abs(Ytest);

%%%%% 1st step:
%%%%%%train an svm for each phase coordinate%%%%%%5
keyboard
rho = 0.2;
Spoints = round(Ltrain1 * rho);

wsvm = zeros(K+1,K);
for k=1:K
[~, Ind] = sort(M1(k,:), 'descend');
Ind = Ind(1:Spoints);
wsvm(:,k) = pegasos_train(M1(:,Ind) , P1(k,Ind));
%bias = wsvm(end);
%wsvm = wsvm(1:end-1);
%Pest(k,:) = sign(wsvm'*M2 + bias);
if mod(k, 64)==0
fprintf('done %d models \n', k)
end

end

%%%% 2nd step.



Rec = pinv(W) * (Mt .* Pest);
Ref = X(:,Ltrain+1:end);

fprintf('naive binary svm prediction is %f \n', normcorrel(Rec, Ref));

%%%%%

if 0
%%%% alternating minimizations %%%%%%% 

iters=16;
Wrk = W * pinv(W); 
Palt = sign(randn(size(Ytest)));
for n=1:iters
Yalt = Mt .* Palt;
Palt = sign(Wrk * Yalt);
if mod(n, 16)==0
fprintf('done %d iterations \n', n)
end
end

Recalt = pinv(W) * (Mt .* Palt);
fprintf('alt minimization prediction is %f \n', normcorrel(Recalt, Ref));

%%%%%%%%
end


%TODO
%%% OK *** proof of concept works pretty well.  93% prediction with binary linear svm, alternating minimization fails completely (7%)
%Now we need to test;
%
%1) n-grams: columns of W form a graph. Coherent atoms should be predicted together
%
%2) sparsity: only train on phases that 'matter' ie use only elements of the training for which atoms are 'critical'. improvement to >96% !!!
%
%3) at test time, similar strategy: only use atoms that produce large enough confidence.
%
% 2 and 3 can be of course combined
%
%4) Replace binary svm with a more powerful machine (eg a neural network?) 
%
%5) Use it in the complex/l2 case by replacing binary classification with multiclass (or even regression on the phase using von-mises)







