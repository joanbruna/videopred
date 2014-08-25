function [Zout1,Zout2,Z]=thinlift(G,A,D,groupsize,M,p,maxiter,dt,lambda,alpha, beta, gamma,verbose,ref);
%min_{X,Z} ||A(X) - G||² + lambda ||X||_* + alpha ||X-DZ||² + beta ||Z||_1 + gamma ||Z||_*

count=0;
g0=16;

[N,K]=size(A);%analysis dictionary where quadratic measurements happen
[N,L]=size(D);%dictionary where x admits a sparse decomposition

DD=D'*D;
Ab=A';

X=dt*init_phase(G,A,p,groupsize);
Z=zeros(L,p);

for k=1:maxiter
    %X update
    [dx,r]=grad_step(Ab,X,G,groupsize,M);
    dx=g0*dx+lambda*X + alpha*(X-D*Z);
    X=X-dt*dx;

    %Z update
    dz = -alpha*(D'*X - DD*Z);   
    dz = dz + gamma*Z;
    Z=soft_th(Z-dt*dz, dt*beta);

    if verbose>0
        if mod(k,verbose)==1
	    count=count+1;
	    errore=abs(sum(ref.*sum(X,2)))^2/(sum(ref.^2)*sum(sum(X,2).^2));
	    t1=(norm(r)^2);
	    t2=(sum(X(:).^2));
	    t3=(norm(X-D*Z)^2);
	    t4=(sum(abs(Z(:))));
            t5=(sum(Z(:).^2));
	    total=t1 + lambda*t2 + alpha*t3 + beta*t4 + gamma*t5;
	    ss=svd(X);
            fprintf('tot %f (%f, %f , %f, %f, %f):::rat %f err %f \n', log(total), log(g0*t1), log(lambda*t2),log(alpha*t3),log(beta*t4),log(gamma*t5), ss(2)/ss(1), errore)
        end      
    end
end

[uu,ss,vv]=svd(X,0);
%diag(ss)
Zout1=uu(:,1)*sqrt(ss(1));
Zout2=sum(X,2);
errore=abs(sum(ref.*Zout1))^2/(sum(ref.^2)*sum(Zout1.^2))

end

function [Y,X]=grad_step(S, Z, G,groupsize, M)

ZZ=S*Z;
X=sum(reshape(sum(ZZ.^2,2),groupsize,M))-G';
Y=((ones(size(S,2),1)*replicate(X',groupsize)').*S')*ZZ;

end

function [Zsnew,X]=grad_step_full(S, Zs, G, groupsize, M, dt, lambda)

ZZ=S*Zs;
X=sum(reshape(sum(ZZ.^2,2),groupsize,M))-G';
YY=((ones(size(S,2),1)*replicate(X',groupsize)').*S')*S;
Znew=Zs*Zs'-dt*YY;

%svd shrink
[u,s,v]=svd(Znew,0);

p=size(Zs,2);
spectr=max(0,s-lambda*dt);
spectr=sqrt(spectr(1:p,1:p));
Zsnew=u(:,1:p)*spectr;

end


function out=init_phase(in, D, p,groupsize)

tutu=(ones(size(D,1),1)*replicate(in,groupsize)');
init=(tutu.*D)*D';
[u,s0,~]=svd(init,0);
for j=1:p
out(:,j) = u(:,j);
end

end

function out=soft_th(Z,th)

out=sign(Z).*max(0,abs(Z)-th);

end

