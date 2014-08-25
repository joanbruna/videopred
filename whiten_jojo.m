function [Xout, im_mean, whitenMatrix, dewhitenMatrix] = whiten_jojo(X)

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


