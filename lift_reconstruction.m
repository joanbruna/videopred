function [out,spect,c1,c2,uo,s,v]=lift_reconstruction(in, D, K,M,L, Drk, Dpinv, niters, ref) 

%this function does the phase recovery in the lifted space
%we try to exploit also the fact that the moduli admit a sparse representation
%by simulataneously estimating these synthesis coefficients

%%min_{Z \geq 0} || A(Dpinv * Z * Dpinv') - b || + lambda ||Z||_* + beta ||Z||_1

t0=1/max(svd(D))^2;
lambda=0.1*t0;
gamma=0.1*t0;
f=80;

t0=.05;
gamma=t0*1e-3;

Z=zeros(K*M);
[u, s, v] = svds(Z, f);

fprintf('precomuting...\n')
Y= precompute_dictionary(D, Dpinv, K);
fprintf('ready \n')
for n=1:niters

%adjust nuclear norm
%Z = u*s*v';
%adjust group L1 norm

if 0
[Z,u,s,v] = prox_l1( u*s*v' - t0*gradient(u,s,v, Dpinv, D, K, in,Y), lambda, K,f);
end
[dZ, c1(n)] = gradient(u,s,v, Dpinv, D, K, in,Y);
[u, s, v,c2(n)] = prox_nuclear( Z - t0*dZ, gamma, f);
out{n}=Dpinv*u(:,1)/sqrt(s(1));
spect{n}=diag(s);
uo{n}=u;
Z = u*s*v';

end


%out=Dpinv*u(:,1);


end


function out=precompute_dictionary(D, Dpinv, K)

[M,N]=size(D);
M=M/K;

for m=1:M
aux=D(1+(m-1)*K:m*K,:);
t1=aux*Dpinv;
out{m}=t1'*t1;
end

end


function [out,cost]=gradient(u , s, v, Dpinv, D, K, ref, Y)

[MM]=size(u,1);
M=MM/K;

out=zeros(MM);

X=(Dpinv*u)*s*(v'*(Dpinv'));
cost=0;
for m=1:M
aux=D(1+(m-1)*K:m*K,:);
tmp = trace(aux*X*aux') - ref(m);
%t1=aux*Dpinv;
out=out+ tmp*Y{m};
cost=cost+tmp^2;
end

end


function [u,s,v,cost]=prox_nuclear(Z, th,f)

[u,s,v]=svds(Z,f);

prevm=max(s(:));
s=max(0,s-th);
%out=u*s*v';

cost=sum(diag(s));
fprintf('svd shrink bef %f aft %f \n', prevm, max(s(:)))


end

function [out,u,s,v]=prox_l1(Z, th, K,f)

[M,M]=size(Z);
out=Z;
for m1=1:K:M
for m2=1:K:M
chunk=Z(m1:m1+K-1,m2:m2+K-1);
nchunk=norm(chunk(:));
out(m1:m1+K-1,m2:m2+K-1)=chunk*max(0,nchunk-th)/nchunk;
end
end
[u,s,v]=svds(out,f);

end

