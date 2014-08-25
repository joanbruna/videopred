function [prederr, A, predX] = linear_prediction(X, I , type)

[N L]=size(X);
L = length(I);
predX = 0*X;

if nargin < 3
type = 0; %regression of the form A*(2Xt-1 - Xt-2)
end

chsize=2^15;

if L > 2*chsize

if type==0
S1bis=zeros(N);
S0bis=zeros(N);
for l=1:chsize:L-chsize
c1= X(:,I(l:l+chsize-1));
c2= 2*X(:,I(l:l+chsize-1)-1) - X(:,I(l:l+chsize-1)-2);
S1bis = S1bis + c1*c2';
S0bis = S0bis + c2*c2';
end
else

S2bis=zeros(N);
S1bis=zeros(N);
S0bis=zeros(N);
for l=1:chsize:L-chsize
c1= X(:,I(l:l+chsize-1));
c2= X(:,I(l:l+chsize-1)-1);
c3= X(:,I(l:l+chsize-1)-2);
S0bis = S0bis + c1*c1';
S1bis = S1bis + c1*c2';
S2bis = S2bis + c1*c3';
end


end


else

%%%%%compute best linear prediction of the form hat(x_t) = A (2x_t-1 - x_t-2)
X0 = X(:,I);
Y  = 2*X(:,I-1) - X(:,I-2);
%X1=0*X;
%X2=0*X;
%X1(:,2:end)=X(:,1:end-1);
%X2(:,2:end)=X1(:,1:end-1);
%Y = 2*X1 - X2;


S1bis = X0*Y';
S0bis = Y*Y';

end

tol=1e-5;
if type==0
A = S1bis * pinv(S0bis,tol);
else
B=[S1bis S2bis];
U = [S0bis S1bis ; S1bis S0bis];
A = B * pinv(U,tol);
end

errt=0;
errb=0;

fprintf('computing pred error now\n')

if L> 2*chsize
if type==0
for l=1:chsize:L-chsize
c1= X(:,I(l:l+chsize-1));
c2= 2*X(:,I(l:l+chsize-1)-1) - X(:,I(l:l+chsize-1)-2);
predX(:,I(l:l+chsize-1)) = A*c2;
c3 = c1 - A*c2;
errt = errt + norm(c3(:))^2;
errb = errb + norm(c1(:))^2;
end
else
for l=1:chsize:L-chsize
c1= X(:,I(l:l+chsize-1));
rien=[X(:,I(l:l+chsize-1)-1) ; X(:,I(l:l+chsize-1)-2)];
predX(:,I(l:l+chsize-1)) = A*rien;
c3 = c1 - A * rien;
errt = errt + norm(c3(:))^2;
errb = errb + norm(c1(:))^2;
end


end

prederr = sqrt(errt/errb);

else

%Xestim = A * Y; 
Xresidu = X - (A)*Y;
prederr=norm(Xresidu(:))/norm(X(:));
end



