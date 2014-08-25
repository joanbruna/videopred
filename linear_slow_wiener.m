function Xslow = linear_slow_wiener(X, lambda)

[N, L]=size(X);

X1=X;
X2=0*X1;
X2(:,2:end)=X1(:,1:end-1);
X2 = X2- X1;

if L > 2^18
chunks=2^17;

S=zeros(N);
SS=zeros(N);
for l=1:chunks:L-chunks
c1= X1(:,l:l+chunks-1);
c2= X2(:,l:l+chunks-1);
S = S + c1*c1';
SS = SS + c2*c2';
end

else

S=X1*X1';
SS=X2*X2';

end

A=S*pinv(S+lambda*SS);

Xslow = A * X;

