function [Xout,scratch] = Ufp_mb(Xin, V, W, alpha, k)

%Xin is size [N, bs]
%V is size [P, bs]
%W is size [M N] 
[P, bs] = size(V);
[N, bs] = size(Xin);

scratch.Y = W*Xin ; 
scratch.norms = sqrt(sum(reshape(scratch.Y, k, P*bs).^2));
scratch.norms = reshape(scratch.norms,P,bs);
scratch.Z = scratch.Y./ replicate(scratch.norms,k);
scratch.UU = scratch.Z .* replicate( V, k) ;
scratch.VV =W' * scratch.UU ;
Xout = Xin + alpha * scratch.VV;

%
%G = W*Xin ; 
%norms = sqrt(sum(reshape(G, k, P*bs).^2));
%norms = reshape(norms,P,bs);
%UU=zeros(P,N,bs);
%for p=1:P
%	slice{p}=W(1+(p-1)*k:p*k,:);
%	UU(p,:,:) = (slice{p}'*G(1+(p-1)*k:p*k,:))./(repmat(norms(p,:),[N 1])) ;
%end
%
%VV(:,1,:)=V;
%scratch.VV = squeeze(sum(repmat(VV,[1 N 1]).*UU));
%%scratch.VV = (sum(repmat(VV,[1 N 1]).*UU));
%if bs==1
%scratch.VV = scratch.VV';
%end
%Xout=Xin + alpha*scratch.VV;
%
%scratch.UU = UU;
%scratch.norms = norms;
%
%

