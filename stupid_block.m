function [empiric,analytic,ref]=stupid_block


N=100;
k=4;
P=40;
bs=4;
eps=1e-7;
S=40;

W=randn(P*k,N);
W=ortho_pools(W,k);

x=randn(N,bs);
[ref,normsref]=FP(W,x,k,P,bs);

IW=randperm(numel(W));
IW=IW(1:S);

SS=200;
II=randperm(numel(ref));
II=II(1:SS);

for n=1:S
Wtmp=W;
Wtmp(IW(n))=W(IW(n))+eps;
outtmp = FP(Wtmp,x, k, P, bs);
empiric(n,:)=(outtmp(II)-ref(II))/eps;
end


for n=1:SS
din=0*ref;
din(II(n))=1;
tmp=BP(W,x,din,ref,normsref,k,P,bs);
analytic(:,n)=tmp(IW);
end


end



function [UU,norms]=FP(W,in, k, P, bs )

%out=W'*W*in;
[N,bs]=size(in);

G = W*in ; 
norms = sqrt(sum(reshape(G, k, P*bs).^2));
norms = reshape(norms,P,bs);
UU=zeros(P,N,bs);
for p=1:P
	slice{p}=W(1+(p-1)*k:p*k,:);
	UU(p,:,:) = (slice{p}'*G(1+(p-1)*k:p*k,:))./(repmat(norms(p,:),[N 1])) ;
end

%out=UU(:);

end

function dW=BP(W, in, vin, UU, norms, k, P, bs)

[M, N]=size(W);

%tmp= in *vin';
%out=W*(tmp+tmp');

dx = zeros(N,bs);
dW = 0*W;

for p=1:P
	slice{p}=W(1+(p-1)*k:p*k,:);
	t1=squeeze(vin(p,:,:));
	t2=squeeze(UU(p,:,:));
	t2aux = repmat(sum(t2.*t1), [N,1]);
	%if bs==1
	%	t2=t2';
	%	t1=t1';
	%end
	dy = (t1 - t2.*t2aux);
	dy=dy./repmat(norms(p,:), [N ,1]);
	dx = dx + slice{p}'*slice{p}*dy;
	t3 = dy * in';
	dW(1+(p-1)*k:p*k,:) = slice{p}*(t3+t3');  
end



end


