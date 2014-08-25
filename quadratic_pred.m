function [Y,Yp] = quadratic_pred(X, options)

%start with a simple verification of our claim.
%we construct the covariance operator 

[N,L]=size(X);

%dimension of subspaces
rho=getoptions(options,'rho',10);
M=getoptions(options,'M',N*rho);

W=randn(M,N);

Y=(W*X).^2;


%now do linear prediction on Y

Yp = linear_prediction(Y);

%linear prediction error in the covariance space
fprintf('prediction error in covariance space %f \n', norm(Y(:)-Yp(:))/norm(Y(:)))




