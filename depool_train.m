
close all;
clear all;

load('/misc/vlgscratch3/LecunGroup/bruna/vocpatches.mat');

[N, L] = size(X);


X= X - repmat(mean(X), size(X,1), 1);


if 1
K=2*N;
W=randn(K,N);

%alternative to zero mean, remove the mean of analysis dictionary.
%lf = 8;
%Wbis = W - repmat(mean(W,2),1,size(W,2));
%W(1:end-lf,:)= Wbis(1:end-lf,:);
else
W = haar_patch_transf_conv(4);
K=size(W,1);
rien=1:K;
aux = (mod(rien,4)==1);
[~,ind]=sort(aux,'ascend');
W = W(ind,:);
end

Y=(W*X);

Ltrain = 500000;

Ytrain = Y(:,1:Ltrain);
Ytest = Y(:,Ltrain+1:end);

P = sign(Ytrain);
M = abs(Ytrain);
Pt = sign(Ytest);
Mt = abs(Ytest);

lowfreq = 0;
if lowfreq > 0
M(end-lowfreq+1:end,:) = Ytrain(end-lowfreq+1:end,:);
Mt(end-lowfreq+1:end,:) = Ytest(end-lowfreq+1:end,:);
end

%%%%%%train an svm for each phase coordinate%%%%%%
if 0
%scale data
[Ms, Mts] = svm_scale_data(M, Mt);
end

rho = 0.25;
Spoints = round(Ltrain * rho);

keyboard;

correl = abs( P * P') ;
[res, indi] = sort(correl(100,:),'descend');
signal = P(100,:) .* P(indi(2),:);

i1 = find(signal==1);
i2 = find(signal==-1);


keyboard;

[uu1,ss1,vv1] = svd( M(:,i1)', 0);
[uu2,ss2,vv2] = svd( M(:,i2)', 0);

%th=0.2;
%d1 = find(diag(ss1) > th*ss1(1));
%d2 = find(diag(ss2) > th*ss2(1));

%Pr1 = vv1(d1,:);
%Pr2 = vv2(d2,:);

Z1 = vv1 * M ;
Z2 = vv2 * M ;

tmp1 = cumsum(Z1.^2);
tmp2 = cumsum(Z2.^2);


for d1=1:4:size(M,1)
for d2=1:4:size(M,1)
scor(d1,d2)=sum((tmp1(d1,:) < tmp2(d2,:)));
end
end




for k=1:K

%model = svmtrain( P(m,:)', Ms');
%Pred(:,m) = svmpredict(  Pt(m,:)', Mts', model);
[~, Ind] = sort(M(k,:), 'descend');
Ind = Ind(1:Spoints);
wsvm = pegasos_train(M(:,Ind) , P(k,Ind));
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







