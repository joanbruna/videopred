
close all;
clear all;


%t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord16.mat');
%t2=load('~/matlab/video_prediction/spamsdictex2.mat');
clear all;
close all;
%
%L=155882;
%X = zeros(32*32*3,L);
%rast=1;
%
%for chunks=1:4
%t1 = load(sprintf('/home/goroshin/Projects/TAE/Data/YouTube_filtered_CN_train.t7_%06d_%06d.mat', (chunks-1)*50000+1,min(L,chunks*50000)));
%X(:,(chunks-1)*50000+1:min(L,chunks*50000)) = reshape(t1.frames,32*32*3,size(t1.frames,4));
%end
%
L=155882;
chsize = 20000;
X = zeros(32*32*8,L);
rast=1;

for chunks=1:8
t1 = load(sprintf('/home/goroshin/Projects/TAE/Data/YouTube_Codes/out%d.mat', chunks));
X(:,(chunks-1)*chsize+1:min(L,chunks*chsize)) = reshape(t1.frames,32*32*8,size(t1.frames,4));
end

%X=t1.Xout;

X = X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]) ;


keyboard

[N, L]=size(X);

X1=X;
X2=0*X1;
X2(:,2:end)=X1(:,1:end-1);
X2 = X2- X1;

if L > 2^18
chunks=2^17;

S=zeros(N);
SS=zeros(N);
for l=1:chunks:L-chunks
c1= X1(:,l:l+chunks-1);
c2= X2(:,l:l+chunks-1);
S = S + c1*c1';
SS = SS + c2*c2';
end

else

S=X1*X1';
SS=X2*X2';

end

C=S*pinv(SS);
C=S;

[uu, ss, vv] = svd(C, 0);

keyboard;

lambda = 2.^[4:5];

cc=colormap('jet');
close all;
figure

for ll=1:1%length(lambda)

if ll> 1
A=S*pinv(S+lambda(ll)*SS);

Xt = A * X;
else
Xt = X;
end

tempth = 100;
nnmax = 20;

%knn_query(Xt', nnmax, tempth);

vl_setup
tree=vl_kdtreebuild(Xt);

[nnid, ndist] = vl_kdtreequery(tree,Xt,Xt, 'NUMNEIGHBORS',nnmax,'MAXCOMPARISONS',250) ;
idx=1:L;

nnid_dist = (abs(double(nnid) - repmat(idx, nnmax, 1))<tempth) ;

for n=2:nnmax

perfo(n-1)=mean(sum(nnid_dist(1:n,:))/n);
perfostd(n-1)=std(sum(nnid_dist(1:n,:))/n);

end

aoc(ll)=sum(perfo);

plot(perfo+perfostd,'Color',cc(round(size(cc,1)*ll/length(lambda)),:));hold on;
plot(perfo,'Color',cc(round(size(cc,1)*ll/length(lambda)),:));hold on;
plot(perfo-perfostd,'Color',cc(round(size(cc,1)*ll/length(lambda)),:));hold on;
leg{ll}=sprintf('lambda %f',lambda(ll)); 

end

legend(leg)

%
%
%X1=t1.Xout;
%X2=0*X1;
%X2(:,2:end)=X1(:,1:end-1);
%X2 = X2- X1;
%
%[N, L] = size(X1);
%
%chunks=2^17;
%
%S=zeros(N);
%SS=zeros(N);
%for l=1:chunks:L-chunks
%c1= X1(:,l:l+chunks-1);
%c2= X2(:,l:l+chunks-1);
%S = S + c1*c1';
%SS = SS + c2*c2';
%end
%
%
%C=S*pinv(SS);
%C=S;
%
%[U, S, V] = svd(C, 0);
%
%
%Vspace = t1.dewhitenMatrix*V;
%
%
