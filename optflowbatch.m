

close all
clear all

t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord32.mat');

II=randperm(size(t1.Xout,2)-2);

L=2000;

I0=II(1:L);
I1=I0+1;
I2=I1+1;

X0 = t1.dewhitenMatrix * t1.Xout(:,I0);
X1 = t1.dewhitenMatrix * t1.Xout(:,I1);
X2 = t1.dewhitenMatrix * t1.Xout(:,I2);

for l=1:L
tempo = optflowpredict(reshape(X0(:,l),32,32), reshape(X1(:,l),32,32) );
err(l)= norm(tempo(:)-X2(:,l))/norm(X2(:,l)); 
Xp(:,l)=tempo(:);
end




