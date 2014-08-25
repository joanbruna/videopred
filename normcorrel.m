function [out,correls]=normcorrel(X, Y)

X = X./(repmat(sqrt(sum(X.^2)),size(X,1),1));
Y = Y./(repmat(sqrt(sum(Y.^2)),size(Y,1),1));

correls=sum(X.*Y);
out=mean(abs(correls));



