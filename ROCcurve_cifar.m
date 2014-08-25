function [FPR, TPR, PPV, AOC] = ROCcurve_cifar(Xtr, Xte, labels_tr, labels_te, options)


%we receive an order dataset. Precision recall curve 
%by varying distance in code space

[N, L] = size(Xtr);
[N, Lt] = size(Xte);
renorm=getoptions(options,'renorm',0);
if renorm
Xtr = Xtr./repmat(max(1e-3,sqrt(sum(Xtr.^2))),[size(Xtr,1) 1]) ;
Xte = Xte./repmat(max(1e-3,sqrt(sum(Xte.^2))),[size(Xte,1) 1]) ;
end

nnmax = getoptions(options,'nnmax',1000);
npoints = getoptions(options,'npoints',10000);

K=kernelizationbis(Xte',Xtr');
[ndist,nnid] = sort(K,2,'ascend');

ndist=ndist(:,1:nnmax);
nnid=nnid(:,1:nnmax);

ndist=ndist';
nnid=nnid';

avi1 = labels_tr(nnid);
avi2 = repmat(labels_te,nnmax,1);

NND= (labels_tr(nnid) == repmat(labels_te,nnmax,1));
NNDD = (labels_tr(nnid) ~= repmat(labels_te,nnmax,1));

snap = NND;

NND=NND(:);
NNDD=NNDD(:);


nnd_tot= sum(NND(:));
nnd_comp = sum(NNDD(:)); % numel(NND) - nnd_tot;

fprintf('totals are %d %d \n', nnd_tot, nnd_comp)
fprintf('1NN performance is %f \n', mean(snap(1,:)))
fprintf('10NN performance is %f \n', mean(mean(snap(1:10,:))))

[~,ind] = sort(ndist(:),'ascend');
TP = cumsum(NND(ind));
FP = cumsum(NNDD(ind));
TPR = TP/nnd_tot;
FPR = FP/nnd_comp;
PPV = TP./max(1,FP+TP);

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

