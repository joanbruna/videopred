function [Xout,X,whitenMatrix, dewhitenMatrix, d] = prepare_charles_data
%function [F,X] = prepare_charles_data


%load all the chunks, for a given patch size, 
% and whiten the data. 



patchsize=32;
numchunks=56;
imsz = 128;
chunklength_t = 64;
D='/misc/vlgscratch3/LecunGroup/bruna/charles_videos/vid075-chunks';

%all data here
F = zeros(imsz,imsz,chunklength_t*numchunks);

for i=1:numchunks,
  s_idx = (chunklength_t*(i-1))+1;
  f_idx = chunklength_t*i;
  F(:,:,s_idx:f_idx) = ...
    read_chunk(D,i,imsz,chunklength_t); 
    
end

%patchify 
border=16;
osampl=8;
counter=0;
for i=border+1:osampl:imsz-max(patchsize,border)
    for j=border+1:osampl:imsz-max(patchsize,border)
        counter=counter+1;
end
end
X=zeros(patchsize*patchsize,counter*size(F,3));

counter=0;
for i=border+1:osampl:imsz-max(patchsize,border)
    for j=border+1:osampl:imsz-max(patchsize,border)
        chunk = F(i:i+patchsize-1,j:j+patchsize-1,:);
        X(:,size(F,3)*counter+1:size(F,3)*(counter+1)) = reshape(chunk,patchsize*patchsize,size(chunk,3));
        counter=counter+1;
end
end

%whiten data.

fractional_vari = 0.01;
cutoff = 1.25;
npatches_white = 100000;

II=randperm(size(X,2));
Z= X(:,II(1:npatches_white));
%substract mean
im_mean = mean(Z,2);
Z = bsxfun(@minus, Z, im_mean);
pixel_variance = var(Z(:));
pixel_noise_variance = pixel_variance * fractional_vari;

ll=100;
npatchtmp = npatches_white/ll;
C=zeros(size(Z,1));
for l=1:ll
chun=Z(:,(l-1)*npatchtmp+1:l*npatchtmp);
C = C + chun*chun';
end
C = C/npatches_white;

clear Z;
[E, D]=eig(C);
[~, sind] = sort(diag(D),'descend');
d = diag(D);
d = d(sind);
E = E(:,sind);
m.imageEigVals = diag(d);
m.imageEigVecs = E;

% determine cutoff:
variance_cutoff = cutoff*pixel_noise_variance;
M = sum(d>variance_cutoff); % number of valid dims
varX = d(1:M);

%factors = real((varX-m.pixel_noise_variance).^(-.5));
factors = real(varX.^(-.5));
E = E(:,1:M);
D = diag(factors);


%% whitening transform
whitenMatrix = D*E';
dewhitenMatrix = E*D^(-1);
zerophaseMatrix = E*D*E';

Xout = bsxfun(@minus, X, im_mean);
Xout = whitenMatrix * Xout;

save('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord32.mat', 'Xout','whitenMatrix','dewhitenMatrix', '-v7.3');


end

function F = read_chunk(dataroot,i,imsz,imszt)
% read_chunk.m - function to read a movie chunk
%
% function F = read_chunk(dataroot,i,imsz,imszt)

filename=sprintf('%s/chunk%d',dataroot,i);
fprintf('%s\n',filename);
fid=fopen(filename,'r','b');
F=reshape(fread(fid,imsz*imsz*imszt,'float'),imsz,imsz,imszt);
fclose(fid);

end






