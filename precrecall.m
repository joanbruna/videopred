function ROCcurve(X, options)

%we receive an order dataset. Precision recall curve 
%by varying distance in code space

X = X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]) ;

vl_setup
tree=vl_kdtreebuild(X);

tempth = getoptions(options,'tempth',5);
nnmax = getoptions(options,'nnmax',2000);
thresholds=0:1e-2:2;

[nnid, ndist] = vl_kdtreequery(tree,X,X, 'NUMNEIGHBORS',nnmax,'MAXCOMPARISONS',250) ;
idx=1:L;
NND = (abs(double(nnid) - repmat(idx, nnmax, 1))<tempth) ;
NNDD = (abs(double(nnid) - repmat(idx, nnmax, 1))>=tempth) ;

nnd_tot= sum(NND(:));
nnd_comp = numel(NND) - nnd_tot;
for s=1:length(thresholds)
I=find(ndist < thresholds(s));
II=find(ndist >= thresholds(s));
TP = sum(NND(I));
FN = sum(NNDD(I));%nnd_tot - TP;
FP = sum(NND(II));
TN = sum(NNDD(II));

TPR(s) = TP/(TP+FN);
FPR(s) = FP/(FP+TN);

end

plot(FPR, TPR);



