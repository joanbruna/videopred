function [Xp] = locallinearpred(X, p)

[N, L]=size(X);

h=ones(1,p);
h2=p/(p-2) * ones(1,p-2);

mu_cur = conv2(X, h2,'same');
mu_past = conv2(X, h, 'same');

tic;
for l=3:L-3
tempo = mu_cur(:,l)*mu_cur(:,l)' - mu_past(:,l)*mu_past(:,l)' + X(:,l-1)*X(:,l-1)';
[U,S,V]=svd(tempo,0);
Xp(:,l+2) = U(:,1);
if mod(l,1000)==999
fprintf('done %d \n',l+1)
toc
tic;
end
end




