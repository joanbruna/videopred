function Xout =  relu_recov(Qin, W, Xinit, options)


%let's start with simple alternating minimization
niters=120;
lr=2e-4;
Xout=Xinit;

if 1

%%gradient descent to start with
Qinn= norm(Qin(:));

for n=1:niters
Y = W*Xout;
G=max(Y,0) - Qin ; 

Xout = Xout - lr * W' * ( (Y>0).* G) ;

%%keyboard
if mod(n,30)==29
fprintf('iter %d err %f \n', n, norm(G(:))/Qinn);
end
%
end
%
elseif 0
%alternting minimizations
Wd=pinv(W);
Y = W*Xout;
Wrk = W*Wd;
Qinsqrt=max(0,Qin);
Qinn = norm(Qinsqrt(:));
for n=1:niters
Y = (Y>0).* Qinsqrt;
Y = Wrk * Y;
%keyboard
if mod(n,30)==29
fprintf('iter %d err %f \n', n, norm(max(0,Y(:))-Qinsqrt(:))/Qinn);
end
end
Xout=Wd*Y;

else




end

%then we can go with phaselift/phasecut versions



