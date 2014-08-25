
close all;
clear all;


t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
t2=load('~/matlab/video_prediction/spamsdictex2.mat');



X=t1.Xout(:,1:500000);

options.wavelet_init = 2;
options.whitenmatrix = t1.whitenMatrix;
options.dewhitenmatrix = t1.dewhitenMatrix;
options.epochs=2;
options.k=2;
%options.M=480;
options.lr=4e-5;

[W, Am, Ap, Wd, erro] = pred_layer_phase_centered(X, options);

if 0
[Ztot, Zptot] = poolingphase(X, W, options.k);
Z0tot = 0*Ztot;
Z0ptot = 0*Zptot;
Z0tot(:,2:end)=Ztot(:,1:end-1);
Z0ptot(:,2:end) = Zptot(:,1:end-1);
Um = Am * (-Z0tot + 2*Ztot);
Up0 = Ap * (-Z0ptot + 2*Zptot);
Up_m = reshape(sqrt(sum(reshape(Up0, options.k, numel(Up0)/options.k).^2)),options.M/options.k, size(Up0,2));
Up = Up0 ./ replicate(Up_m, options.k);
Umm = replicate(Um, options.k);

Q = Wd * (Umm.*Up);
Xr=0*X;
Xr(:,1:end-1) = X(:,2:end);

II=randperm(size(Up0,2));
II=II(1:400);

chunk = Xr(:,II);
chunkp = Q(:,II);
end










