
close all;
clear all;

options.L=400000;
options.epochs=2;
options.k=4;
options.M=480;
options.lr=4e-5;


X= generate_jitter_data(options);
X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);

options.wavelet_init = 2;
[W, Am, Ap, Wd, erro] = pred_layer_phase_centered(X, options);

if 1
[Ztot, Zptot] = poolingphase(X, W, options.k);
Z0tot = 0*Ztot;
Z0ptot = 0*Zptot;
Z0tot(:,2:end)=Ztot(:,1:end-1);
Z0ptot(:,2:end) = Zptot(:,1:end-1);
Z1tot = 0*Ztot;
Z1ptot = 0*Zptot;
Z1tot(:,1:end-1)=Ztot(:,2:end);
Z1ptot(:,1:end-1) = Zptot(:,2:end);
Um = Am * (Z0tot + Z1tot);
Up0 = Ap * (Z0ptot + Z1ptot);
Up_m = reshape(sqrt(sum(reshape(Up0, options.k, numel(Up0)/options.k).^2)),options.M/options.k, size(Up0,2));
Up = Up0 ./ replicate(Up_m, options.k);
Umm = replicate(Um, options.k);

Q = Wd * (Umm.*Up);
%Xr=0*X;
%Xr(:,1:end-1) = X(:,2:end);

II=randperm(size(Up0,2));
II=II(1:400);

chunk = X(:,II);
chunkp = Q(:,II);
end










