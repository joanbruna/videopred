
if 1
close all;
clear all;


t1 = load('/scratch/bruna/youtube_data.mat');

end

X=t1.X;

options.wavelet_init = 2;
options.epochs=1;
options.k=2;
options.M=1024;
options.lr=4e-3;
%options.lr=6e-4;
%options.kappa=2e-1;
options.reconstr=0;

[ W, W0, err_snap] = nuclear_learning(X, options);









