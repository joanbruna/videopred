function I=get_legal_frames(X, options)

time_memory=getoptions(options,'time_memory',2);
outliers_fract= getoptions(options,'outliers_time',0.01);
[N, L]=size(X);

%identify transitions
O=round(L*outliers_fract);

Xtmp = 0*X;
Xtmp(:,2:end)=X(:,1:end-1);
Xtmp = X - Xtmp;
normes=sqrt(sum(Xtmp.^2));
normes1=0*normes;
normes2=0*normes;
normes1(2:end)=normes(1:end-1);
normes2(1:end-1)=normes(2:end);
normbis = normes./(normes1+normes2);
%border effect at the beginning 
normbis(1:2)=max(normbis);
normbis(end-1:end)=max(normbis);
[val,pos]=sort(normbis,'descend');
legals=pos(O+1:end);

%r=randperm(length(legals));
%I=legals(r);

I=legals;




