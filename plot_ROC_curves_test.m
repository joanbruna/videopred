clear all;
close all;

L=20000;
options.nnmax=2000;
options.tempth=3;

rast=1;

%GSC codes
X = zeros(32*32*8,L);

for type=1:6

t1 = load(sprintf('/home/goroshin/Projects/TAE/Results/Experiments/codes/GSCBatchNIPS/GSC%d/test.mat', type));
X = reshape(t1.frames,32*32*8,L);

[fpr{rast}, tpr{rast}, ppv{rast}, auc(rast)] = ROCcurve(X, options);
leg{rast}=sprintf('GSC%d',type);rast=rast+1;

end

%SF codes
X = zeros(32*32*8,L);

for type=1:6

t1 = load(sprintf('/home/goroshin/Projects/TAE/Results/Experiments/codes/SFBatchNIPS/SF%d/test.mat', type));
X = reshape(t1.frames,32*32*8,L);

[fpr{rast}, tpr{rast}, ppv{rast}, auc(rast)] = ROCcurve(X, options);
leg{rast}=sprintf('SF%d',type);rast=rast+1;

end

%input data
X=zeros(32*32*3,L);
t1 = load('/home/goroshin/Projects/TAE/Data/YouTube_filtered_CN_test.t7.mat');
X = reshape(squeeze(t1.frames(:,:,:,1,:)),32*32*3,L);

[fpr{rast}, tpr{rast}, ppv{rast}, auc(rast)] = ROCcurve(X, options);
leg{rast}=sprintf('input',type);rast=rast+1;


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
plot(tpr{rr}(2:end),ppv{rr}(2:end), 'Color',cc(rr*step,:));hold on
end
legend(leg)


