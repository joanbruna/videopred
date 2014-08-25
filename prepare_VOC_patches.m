
close all;
clear all;

cd /misc/vlgscratch1/LecunGroup/aszlam/VOC/VOCdevkit/VOC2011/JPEGImages

npi=300
nunof=8000000;
dd=dir;
count=0;
psize=16;
cnormalize=0;
U=zeros(psize,psize,nunof);

for k=3:length(dd);
    a=imread(dd(k).name,'jpg');
    a=double(rgb2gray(imresize(a,.5)));
    [hh ww]=size(a);
    [id nbid]=borderfind(hh,ww,psize/2);
    if length(nbid)>npi
    rid=randperm(length(nbid));
    
    for j=1:npi
        count=count+1;
        idx=paex(hh,ww,nbid(rid(j)),psize);
        u=a(idx);
        U(:,:,count)=reshape(u,psize,psize);
        if count>nunof
            break
        end
        if mod(count,10000)==1
            ta=a;
            ta(idx)=ta(idx)+60;
            figure(5);imagesc(ta);colormap(gray);
        end
    end
    end
    if count>nunof
            break
        end
    
end


s=sqrt(squeeze(sum(sum(U.^2,1),2)));
uid=find(s>100);
U=U(:,:,uid);
U=reshape(U,256,size(U,3));
mu=mean(U);
st=sum((U-ones(256,1)*mu).^2);
X=U(:,st>.5e6);
N=size(X,2);
%X=reshape(X,16,16,N);

save('/misc/vlgscratch3/LecunGroup/bruna/vocpatches.mat','X','-v7.3');

%cd '/home/bruna/matlab/video_prediction'






