
close all;
clear all;

load('/misc/vlgscratch3/LecunGroup/bruna/vocpatches.mat');

[N, L] = size(X);

K=2*N;
W=randn(K,N);


Y=(W*X);

Ltrain = 500000;

Ytrain = Y(:,1:Ltrain);
Ytest = Y(:,Ltrain+1:end);

P = sign(Ytrain);
M = abs(Ytrain);
Pt = sign(Ytest);
Mt = abs(Ytest);

%%%%%%train an svm for each phase coordinate%%%%%%5
if 0
%scale data
[Ms, Mts] = svm_scale_data(M, Mt);
end

keyboard

for k=1:K

%model = svmtrain( P(m,:)', Ms');
%Pred(:,m) = svmpredict(  Pt(m,:)', Mts', model);
wsvm = pegasos_train(M , P(k,:));
bias = wsvm(end);
wsvm = wsvm(1:end-1);
Pest(k,:) = sign(wsvm'*Mt + bias);

if mod(k, 64)==0
fprintf('done %d models \n', k)
end

end


Rec = pinv(W) * (Mt .* Pest);
Ref = X(:,Ltrain+1:end);

fprintf('naive binary svm prediction is %f \n', normcorrel(Rec, Ref));

%%%%%

%%%% alternating minimizations %%%%%%% 

iters=16;
Wrk = W * pinv(W); 
%Palt = sign(randn(size(Ytest)));
Palt = Pest;
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


%TODO
%%% OK *** proof of concept works pretty well.  93% prediction with binary linear svm, alternating minimization fails completely (7%)
%Now we need to test;
%
%1) n-grams: columns of W form a graph. Coherent atoms should be predicted together
%
%2) sparsity: only train on phases that 'matter' ie use only elements of the training for which atoms are 'critical'. 
%
%3) at test time, similar strategy: only use atoms that produce large enough confidence.
%
% 2 and 3 can be of course combined
%
%
%4) Replace binary svm with a more powerful machine (eg a neural network?) 
