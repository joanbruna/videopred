
close all;
clear all;


tmp = load('~/matlab/graph_cnn/mnisttrain.mat');
N=28*28;
L=60000;
X=zeros(N,L);
Y=zeros(1,L);
for i=1:L
X(:,i) = tmp.data{i}.X(:);
Y(i)=find(tmp.data{i}.Y);
end

[Y,II]=sort(Y);
X=X(:,II);

X0=X';
[U, S, V] = svd(X0,0);
%X0 = U * S * V';
%X = V * S * U';
kk=256;
X=U(:,1:kk);
X=X';

X=X./repmat(sqrt(sum(X.^2)),[kk 1]);


options.wavelet_init = 2;
options.epochs=20; %number of passes through the data
options.k=2; %groups (k=2 means 'complex' wavelets)
options.lr=4e-3;
options.reconstr=0;

[ W, W0, err_snap] = nuclear_learning_discrim(X, Y, options);

dewhiten = V(:,1:kk) * S(1:kk,1:kk);
aviam = W * dewhiten' ;









