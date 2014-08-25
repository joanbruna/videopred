function [dx, dv, dW,dalpha, tout] = Ubp_mb(x, v, W, alpha, din, k, scratch) 
%this function backpropagates through the phase predictior

%din is size [N,bs]
%x is size [N,bs]
%v is size [P,bs]
%W is size [M N]

%scratch.UU is size [M bs]
%scratch.Z is size [M bs]
%scratch.Y is size [M bs]
%scratch.norms is size [P bs]

%k: size of pools
[M, N]=size(W);
[N, bs]=size(x);
P=M/k; %number of pools

dW0 =  din * scratch.UU';
dUU = W * din ;
dZ = dUU .* replicate(v,k);
dv = reshape(sum(reshape(dUU .* scratch.Z, k , P*bs)),P,bs);
dY = (dZ - scratch.Z .* reshape(repmat( sum(reshape( scratch.Z .* dZ, k, P*bs)), [k 1]),M, bs))./replicate(scratch.norms,k);
dx = W' * dY;
dW = dY * x' + dW0';

dx = alpha * dx + din;
dW = alpha * dW;
dv = alpha * dv;
dalpha = sum(scratch.VV(:).*din(:));

if 0

%G = W*x ; 
%norms = sqrt(sum(reshape(G, k, P*bs).^2));
%norms = reshape(norms,P,bs);
%UU=zeros(P,N,bs);
%for p=1:P
%	slice{p}=W(1+(p-1)*k:p*k,:);
%	UU(p,:,:) = (slice{p}'*G(1+(p-1)*k:p*k,:))./(repmat(norms(p,:),[N 1])) ;
%end
norms = scratch.norms;
UU=scratch.UU;


tmp(1,:,:)=din;
tmp=repmat(tmp,[P 1 1]);
dv =sum(UU.*tmp,2);

dUU=zeros(size(UU));

%if 0
%vt(:,1,:)=v;
%vt=repmat(vt,[1 N 1]);
%dt(1,:,:)=din;
%dt=repmat(dt,[P 1 1]);
%dUU=vt.*dt;
%else
%this implementation is faster for bs approx 16
for b=1:bs
dUU(:,:,b) = v(:,b)*(din(:,b)');
end
%end

dx = zeros(N,bs);
dW = 0*W;

for p=1:P
	slice{p}=W(1+(p-1)*k:p*k,:);
	t1=squeeze(dUU(p,:,:));
	t2=squeeze(UU(p,:,:));
	t2aux = repmat(sum(t2.*t1), [N,1]);
	%if bs==1
	%	t2=t2';
	%	t1=t1';
	%end
	dy = (t1 - t2.*t2aux);
	dy=dy./repmat(norms(p,:), [N ,1]);
	dx = dx + slice{p}'*slice{p}*dy;
	t3 = dy * x';
	dW(1+(p-1)*k:p*k,:) = slice{p}*(t3+t3');  
end

dx = alpha * dx + din;
dW = alpha * dW;
dv = alpha * dv;
dalpha = sum(scratch.VV(:).*din(:));

dv = squeeze(dv);

%end
tout=toc;

end

