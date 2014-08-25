function [W, A, alpha]=init_pooling(X, M, k, options)

%let's start with simple K-Means
[N, L]=size(X);
P=M/k;

wavelet_init = getoptions(options,'wavelet_init',0);

if wavelet_init==1

whiten_matrix = getoptions(options,'whitenmatrix',eye(size(X,1)));
%construct random wavelets of scale 1 and 2 and different orientations
N=16;
opto.L=6;
opto.J=2;
opto=configure_wavelet_options(opto);
opto.filts = opto.filter_bank_name([N N],opto);
template=zeros(N);
for p=1:P
%draw random spatial location, orientation and scale
template=0*template;
template(max(1,round(N*N*rand)))=1;
ori=ceil(opto.L*rand);
sc=ceil(opto.J*rand);
tmp = ifft2( fft2(template).*opto.filts.psi{1}{sc}{ori});
W(2*(p-1)+1,:)= real(tmp(:));
W(2*(p-1)+2,:)= imag(tmp(:));
end

W = W * whiten_matrix' ; 

elseif wavelet_init==2
%random init
W=randn(M,N);
W=ortho_pools(W,k);
W= 30*W/norm(W(:));

else
if isfield(options,'initdictionary')
cents = options.initdictionary;
else
size(X);
P;
[~,cents]=litekmeans_orig(X, P);
end


ncents=sqrt(sum(cents.^2));
cents=cents./repmat(ncents,[N 1]);
eps=0.05/sqrt(N);

W=zeros(M,N);
for p=1:P
W(1+(p-1)*k:p*k,:) = repmat(cents(:,p)',[k 1]) + eps * randn(k,N);
end

end


%W = ortho_pools(W, k);
A=eye(P);
alpha = -0.25;



