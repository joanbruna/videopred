function [dx, dv, dW] = Ubp(x, v, W, din, k) 
%this function backpropagates through the phase predictior

%din is size [N,1]
%x is size [N,1]
%v is size [P,1]
%W is size [M N]

%k: size of pools
[M, N]=size(W);
P=M/k; %number of pools

G = W*x ; 
norms = sqrt(sum(reshape(G, k, P).^2));
UU=zeros(P,N);
for p=1:P
	slice{p}=W(1+(p-1)*k:p*k,:);
	UU(p,:) = (slice{p}'*G(1+(p-1)*k:p*k))/norms(p);
end

dv = UU * din; 
dUU = v * din';
dx = zeros(1,N);

for p=1:P
	dy = (dUU(p,:) - UU(p,:)*sum(UU(p,:).*dUU(p,:)))/norms(p);
	dx = dx + dy*slice{p}'*slice{p};
	dW(:,1+(p-1)*k:p*k) = dy'*x'*slice{p}' + x*dy*slice{p}';
end

dx=dx';






