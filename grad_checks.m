
close all
clear all
%%% check the gradient implementation of the modules

N=100;
k=4;
P=40;
M=k*P;
bs=4;


X=randn(N,bs);
V=randn(P,bs);
W=randn(M,N);
W=ortho_pools(W,k);
alpha=0.1;



[ref,scratch] = Ufp_mb(X,V,W,alpha,k);

SS=100;
II=randperm(numel(ref));
II=II(1:SS);
ref0=ref(II);


S=100;
eps=1e-7;

%perturbations wrt X, V, and W
IX=randperm(numel(X));
IX=IX(1:S);
IV=randperm(numel(V));
IV=IV(1:S);
IW=randperm(numel(W));
IW=IW(1:S);

for i=1:S
Xbis=X;
Xbis(IX(i))=X(IX(i))+eps;
rien = Ufp_mb(Xbis, V, W, alpha, k);
empiricX(i,:)=(rien(II) - ref0)/eps;

Vbis=V;
Vbis(IV(i))=V(IV(i))+eps;
rien = Ufp_mb(X, Vbis, W, alpha, k);
empiricV(i,:)=(rien(II) - ref0)/eps;

Wbis=W;
Wbis(IW(i))=W(IW(i))+eps;
rien = Ufp_mb(X, V, Wbis, alpha, k);
empiricW(i,:)=(rien(II) - ref0)/eps;
end

for i=1:SS
din=0*ref;
din(II(i))=1;
[G, dV0, dW0,dalpha0]=Ubp_mb(X,V,W,alpha,din,k,scratch);
analyticX(:,i) = G(IX);
analyticV(:,i) = dV0(IV);
analyticW(:,i) = dW0(IW);
end

empiricX = empiricX(:)/norm(empiricX(:));
empiricV = empiricV(:)/norm(empiricV(:));
empiricW = empiricW(:)/norm(empiricW(:));
analyticX = analyticX(:)/norm(analyticX(:));
analyticV = analyticV(:)/norm(analyticV(:));
analyticW = analyticW(:)/norm(analyticW(:));

fprintf('error wrt X is %f \n',norm(empiricX - analyticX))
fprintf('error wrt V is %f \n',norm(empiricV - analyticV))
fprintf('error wrt W is %f \n',norm(empiricW - analyticW))

