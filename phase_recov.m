function Xout =  phase_recov(Qin, W, Xinit, options)


%let's start with simple alternating minimization
niters=200;
lr=1e-8;
Xout=Xinit;

if 0

%%gradient descent to start with
Qinn= norm(Qin(:));

for n=1:niters
Y = W*Xout;
G=(Y).^2 - Qin ; 

Xout = Xout - lr * W' * ( Y.* G) ;

%%keyboard
if mod(n,30)==29
fprintf('iter %d err %f \n', n, norm(G(:))/Qinn);
end
%
end
%
else
%alternating minimizations
Wd=pinv(W);
Y = W*Xout;
Wrk = W*Wd;
Qinsqrt=sqrt(abs(Qin));
Qinn = norm(Qinsqrt(:));
for n=1:niters
Y = sign(Y).* Qinsqrt;
Y = Wrk * Y;
%keyboard
if mod(n,50)==49
fprintf('iter %d err %f \n', n+1, norm(abs(Y(:))-Qinsqrt(:))/Qinn);
end
end
Xout=Wd*Y;

end

%then we can go with phaselift/phasecut versions
%TODO phaselift is necessary: there is still hope, in the sense that alt minimizations get stuck at energy levels higher than 
%the one given by the good prediction. THere is thus still hope that with a better minimization we get closer to that predictor. 


