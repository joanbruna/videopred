function [FPR, TPR, PPV, AOC] = ROCcurve(X, options)

%we receive an order dataset. Precision recall curve 
%by varying distance in code space

[N, L] = size(X);
renorm=getoptions(options,'renorm',1);
if renorm
X = X./repmat(max(1e-3,sqrt(sum(X.^2))),[size(X,1) 1]) ;
end


tempth = getoptions(options,'tempth',5);
nnmax = getoptions(options,'nnmax',1000);
npoints = getoptions(options,'npoints',10000);
queries = getoptions(options,'queries',5000);
II=randperm(L);
II=II(1:queries);
Xq=X(:,II);

K=kernelizationbis(Xq',X');
[ndist,nnid] = sort(K,2,'ascend');

ndist=ndist(:,2:nnmax+1);
nnid=nnid(:,2:nnmax+1);

ndist=ndist';
nnid=nnid';

NND = (abs(double(nnid) - repmat(II, nnmax, 1))<tempth) ;
NNDD = (abs(double(nnid) - repmat(II, nnmax, 1))>=tempth) ;

NND=NND(:);
NNDD=NNDD(:);


nnd_tot= sum(NND(:));
nnd_comp = sum(NNDD(:)); % numel(NND) - nnd_tot;

[~,ind] = sort(ndist(:),'ascend');
TP = cumsum(NND(ind));
FP = cumsum(NNDD(ind));
TPR = TP/nnd_tot;
PPV = TP./max(1,FP+TP);
FPR = FP/nnd_comp;
AOC = sum( TPR.*gradient(FPR)) ;

downsample=round(length(ind)/npoints);
TPR = TPR(1:downsample:end);
FPR = FPR(1:downsample:end);
PPV = PPV(1:downsample:end);

end

function [ker,kern]=kernelizationbis(data,databis)

[L,N]=size(data);
[M,N]=size(databis);

norms=sum(data.^2,2)*ones(1,M);
normsbis=sum(databis.^2,2)*ones(1,L);
ker=norms+normsbis'-2*data*(databis');

end

