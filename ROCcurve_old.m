function [FPR, TPR, ndist] = ROCcurve(X, options)

%we receive an order dataset. Precision recall curve 
%by varying distance in code space

[N, L] = size(X);
renorm=getoptions(options,'renorm',1);
if renorm
X = X./repmat(max(1,sqrt(sum(X.^2))),[size(X,1) 1]) ;
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

if 0
vl_setup
tree=vl_kdtreebuild(X);
[nnid, ndist] = vl_kdtreequery(tree,X,Xq, 'NUMNEIGHBORS',nnmax,'MAXCOMPARISONS',0) ;

nnid=nnid(2:end,:);
ndist=ndist(2:end,:);

ss=sum(isnan(ndist(:)));
if ss>0
fprintf('repeating nn with more comparisons..\n')
[nnid, ndist] = vl_kdtreequery(tree,X,Xq, 'NUMNEIGHBORS',nnmax) ;
else
fprintf('neighbs are OK \n');
end

end

NND = (abs(double(nnid) - repmat(II, nnmax, 1))<tempth) ;
NNDD = (abs(double(nnid) - repmat(II, nnmax, 1))>=tempth) ;

NND=NND(:);
NNDD=NNDD(:);


nnd_tot= sum(NND(:));
nnd_comp = sum(NNDD(:)); % numel(NND) - nnd_tot;

[~,ind] = sort(ndist(:),'ascend');
%I = ind(1:downsample:end);
TP = cumsum(NND(ind));
%FN = nnd_tot - TP;
FP = cumsum(NNDD(ind));
%TN = nnd_comp - FP;
TPR = TP/nnd_tot;%(TP+FN);
FPR = FP/nnd_comp;%./(FP+TN);

downsample=round(length(ind)/npoints);
TPR = TPR(1:downsample:end);
FPR = FPR(1:downsample:end);

%
%
%for s=1:length(ind)
%I=ind(1:s);
%II=ind(s+1:end);
%TP = sum(NND(I));
%FN = sum(NND(II));%nnd_tot - TP;
%FP = sum(NNDD(I));
%TN = sum(NNDD(II));
%
%TPR(s) = TP/(TP+FN);
%FPR(s) = FP/(FP+TN);
%
%end
%

%for s=1:length(thresholds)
%I=find(ndist < thresholds(s));
%II=find(ndist >= thresholds(s));
%TP = sum(NND(I));
%FN = sum(NND(II));%nnd_tot - TP;
%FP = sum(NNDD(I));
%TN = sum(NNDD(II));
%
%TPR(s) = TP/(TP+FN);
%FPR(s) = FP/(FP+TN);
%
%end
%
%plot(FPR, TPR);



