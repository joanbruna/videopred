function Xout =  phase_recov_lift(Qin, W, lambda, r);

%we solve min_X || f(X) - Qin ||_2^2 + lambda * || X ||_* 
%and output x = argmin_x || X - xx^T|| 

%W is size [M, N]
[M, L] = size(Qin);
[M, N] = size(W);
niters=100;
verb=50;
gamma=2e-2; %1/max(svd(W))^2

D = prepare_dictionary(W);

%I don't know how to get rid of the example loop for the moment.
for l=1:L

%X = zeros(N);
ref = Qin(:,l);
nref = norm(ref(:));
[LL, RR]=init_matrices(ref', D, r);

for n=1:niters

tmp = (W * LL)*RR ;
tmp = tmp .* W;
tmp = sum(tmp,2) - ref;
if(mod(n,verb)==verb-1)
fprintf('err is %f \n', norm(tmp(:))/nref )
end

dX = reshape(sum(repmat(tmp', N*N, 1).* D,2),N,N); 
dLL = dX * RR' + lambda * LL;
dRR = LL' * dX + lambda * RR;

LL = LL - gamma * dLL ;
RR = RR - gamma * dRR;

end
[uu,ss,vv]=svd(LL*RR,0);
Xout(:,l)=uu(:,1);
%keyboard;
end

end


function [LL,RR]=init_matrices(ref, D, r)

[NN,M]=size(D);
N=sqrt(NN);
%gamma = 1e-6;

cosa = reshape(sum(repmat(ref,N*N,1).*D,2),N,N);
[u, s, v]= svd(cosa,0);
s=1e-4* s/sum(s(:));
mim=s(1:r,1:r).^(1/2);
LL=u(:,1:r)*mim;
RR=v(:,1:r)*mim;
RR=RR';


end

function D=prepare_dictionary(W)

[M, N]=size(W);
D=zeros(N,N,M);

for m=1:M
D(:,:,m) = W(m,:)'* W(m,:);
end
D=reshape(D,N*N, M);
end
