clear all;
close all;

L=155882;
options.nnmax=2000;

%input data
X = zeros(32*32*3,L);

for chunks=1:4
t1 = load(sprintf('/home/goroshin/Projects/TAE/Data/YouTube_filtered_CN_train.t7_%06d_%06d.mat', (chunks-1)*50000+1,min(L,chunks*50000)));
X(:,(chunks-1)*50000+1:min(L,chunks*50000)) = reshape(t1.frames,32*32*3,size(t1.frames,4));
end


save('/scratch/bruna/youtube_data.mat','X', '-v7.3');

return;

[fpr, tpr] = ROCcurve(X, options);
figure;
plot(fpr,tpr);hold on
leg{1}='input';

keyboard

%codes

chsize = 20000;
X = zeros(32*32*8,L);

for chunks=1:8
t1 = load(sprintf('/home/goroshin/Projects/TAE/Data/YouTube_Codes/out%d.mat', chunks));
X(:,(chunks-1)*chsize+1:min(L,chunks*chsize)) = reshape(t1.frames,32*32*8,size(t1.frames,4));
end

[fpr, tpr] = ROCcurve(X, options);
plot(fpr,tpr,'r');hold on
leg{2}='codes';

%smoothed input data
X = zeros(32*32*3,L);

for chunks=1:4
t1 = load(sprintf('/home/goroshin/Projects/TAE/Data/YouTube_filtered_CN_train.t7_%06d_%06d.mat', (chunks-1)*50000+1,min(L,chunks*50000)));
X(:,(chunks-1)*50000+1:min(L,chunks*50000)) = reshape(t1.frames,32*32*3,size(t1.frames,4));
end
lambda = 1;
Xl = linear_slow_wiener(X, lambda);

options.renorm=1;
[fpr, tpr] = ROCcurve(Xl, options);
%figure;
plot(fpr,tpr,'g');hold on
leg{3}='smoothed input';

legend(leg);



