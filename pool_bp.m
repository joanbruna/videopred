function dWout = pool_bp(dXin, W, X, Z0, Z, k)

%G = W*Xin ; 
%norms = sqrt(sum(reshape(G, k, P*bs).^2));
%norms = reshape(norms,P,bs);
[P, bs]=size(dXin);
[M, N]= size(W);
dWout=0*W;

dWout = (Z0 .* replicate(dXin./Z,k)) * X';

%UU=zeros(P,N,bs);
%for p=1:P
%	slice{p}=W(1+(p-1)*k:p*k,:);
%	dWout(1+(p-1)*k,k*p,:) = Z0(1+(p-1)*k,p*k,:)*X'./(repmat(Z(p,:),[k 1]));
%	UU(p,:,:) = (slice{p}'*ZZ(1+(p-1)*k:p*k,:))./(repmat(Z(p,:),[N 1])) ;
%end
%VV(:,1,:)=dXin;
%dWout=squeeze(sum(repmat(VV,[1 N 1]).*UU));

%size(dWout)
%size(W)



