clear all;
close all;

Ltrain=50000;
Ltest=10000;
chsize=20000;
options.nnmax=2000;
options.renorm=1;

rast=1;

%space
Xtr = zeros(32*32*3,Ltrain);
Xte = zeros(32*32*3,Ltest);

%load labels
labels_train = zeros(Ltrain,1);
for i=1:3
t1 = load(sprintf('/home/goroshin/Projects/TAE/Results/Experiments/codes/CIFAR/Xtrain%d.mat', i));
labels_train(1+(i-1)*chsize:min(Ltrain,i*chsize))=t1.labels;
Xtr(:,(i-1)*chsize+1:min(Ltrain,i*chsize)) = reshape(t1.frames,32*32*3,size(t1.frames,4));
end
clear t1;
t1 = load('/home/goroshin/Projects/TAE/Results/Experiments/codes/CIFAR/Xtest.mat');
Xte = reshape(t1.frames,32*32*3,Ltest);
labels_test = t1.labels;

[fpr{rast}, tpr{rast}, ppv{rast}, aoc(rast)] = ROCcurve_cifar(Xtr, Xte, labels_train', labels_test', options);
leg{rast}='pixels';rast=rast+1;

%GSC codes
Xtr = zeros(32*32*8,Ltrain);
Xte = zeros(32*32*8,Ltest);
for type=1:6

%train
for chunk=1:3
t1 = load(sprintf('/home/goroshin/Projects/TAE/Results/Experiments/codes/CIFAR/GSCBatchNIPS/GSC%d/train%d.mat', type,chunk));
Xtr(:,(chunk-1)*chsize+1:min(Ltrain,chunk*chsize)) = reshape(t1.frames,32*32*8,size(t1.frames,4));
end
%test
t1 = load(sprintf('/home/goroshin/Projects/TAE/Results/Experiments/codes/CIFAR/GSCBatchNIPS/GSC%d/test.mat', type));
Xte = reshape(t1.frames,32*32*8,Ltest);


[fpr{rast}, tpr{rast}, ppv{rast}, aoc(rast)] = ROCcurve_cifar(Xtr, Xte, labels_train', labels_test', options);
leg{rast}=sprintf('GSC%d',type);rast=rast+1;

end

keyboard

%SF codes
Xtr = zeros(32*32*8,Ltrain);
Xte = zeros(32*32*8,Ltest);
for type=1:6

%train
for chunk=1:3
t1 = load(sprintf('/home/goroshin/Projects/TAE/Results/Experiments/codes/CIFAR/SFBatchNIPS/SF%d/train%d.mat', type,chunk));
Xtr(:,(chunk-1)*chsize+1:min(Ltrain,chunk*chsize)) = reshape(t1.frames,32*32*8,size(t1.frames,4));
end
%test
t1 = load(sprintf('/home/goroshin/Projects/TAE/Results/Experiments/codes/CIFAR/SFBatchNIPS/SF%d/test.mat', type));
Xte = reshape(t1.frames,32*32*8,Ltest);


[fpr{rast}, tpr{rast}, ppv{rast}, aoc(rast)] = ROCcurve_cifar(Xtr, Xte, labels_train', labels_test', options);
leg{rast}=sprintf('SF%d',type);rast=rast+1;

end

keyboard

cc=colormap('hsv');
cn=size(cc,1);
step=4;%round(cn/rast);

figure
for rr=1:rast-1
plot(fpr{rr},tpr{rr},'Color',cc(rr*step,:));hold on
end
legend(leg)

figure
for rr=1:rast-1
plot(tpr{rr}(1:end),ppv{rr}(1:end), 'Color',cc(rr*step,:));hold on
end
legend(leg)


figure
plot(aoc(2:7));
title('AOC for GSC')

figure
plot(aoc(8:end))
title('AOC for SF')



%
%%input data
%X = zeros(32*32*3,L);
%
%for chunks=1:4
%t1 = load(sprintf('/home/goroshin/Projects/TAE/Data/YouTube_filtered_CN_train.t7_%06d_%06d.mat', (chunks-1)*50000+1,min(L,chunks*50000)));
%X(:,(chunks-1)*50000+1:min(L,chunks*50000)) = reshape(t1.frames,32*32*3,size(t1.frames,4));
%end
%
%
%[fpr, tpr] = ROCcurve(X, options);
%figure;
%plot(fpr,tpr);hold on
%leg{1}='input';
%%smoothed input data
%X = zeros(32*32*3,L);
%
%for chunks=1:4
%t1 = load(sprintf('/home/goroshin/Projects/TAE/Data/YouTube_filtered_CN_train.t7_%06d_%06d.mat', (chunks-1)*50000+1,min(L,chunks*50000)));
%X(:,(chunks-1)*50000+1:min(L,chunks*50000)) = reshape(t1.frames,32*32*3,size(t1.frames,4));
%end
%lambda = 1;
%Xl = linear_slow_wiener(X, lambda);
%
%options.renorm=1;
%[fpr, tpr] = ROCcurve(Xl, options);
%%figure;
%plot(fpr,tpr,'g');hold on
%leg{3}='smoothed input';
%
%legend(leg);
%
%
%
