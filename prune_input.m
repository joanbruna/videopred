function out=prune_input(in, n)

[N1,N2] = size(in);

II=randperm(N2/2);
I=II(1:n/2);
J1=2*I - 1;
J2=2*I;
J=[J1;J2];
J=J(:);

out=in(:,J);



