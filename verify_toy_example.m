close all;
clear all;

cd('/misc/vlgscratch2/LecunGroup/goroshin/rotate_toyplane')


X=zeros(90,90,96*96);

for i=1:90
for j=1:90

tmp=rgb2gray(imread(sprintf('toyplane_%02d_%02d.png',i, j)));
X(i,j,:)=tmp(:);

end
end

Xb=X(:,:);

keyboard

[u, s, v] = svds(Xb,3);

scatter3(

