function filters=prepare_filters_2d(N, options)

options.wavelet_family='cauchy';
options=configure_wavelet_options(options);

tempo=options.filter_bank_name([N N], options);
rectified = getoptions(options,'rectified',0);

J=size(tempo.psi{1},2);
L=size(tempo.psi{1}{1},2);

for j=1:J
for l=1:L
if rectified
filters{1}.psi{j}{l} = (fft2(real(ifft2(tempo.psi{1}{j}{l}))));
filters{2}.psi{j}{l} = (fft2(real(ifft2(tempo.psi{1}{j}{l}))));
else
filters{1}.psi{j}{l} = tempo.psi{1}{j}{l};
filters{2}.psi{j}{l} = tempo.psi{1}{j}{l};
end
%filters{1}.rpsi{j}{l} = conj(fft2(real(ifft2(tempo.psi{1}{j}{l}))));
%filters{2}.rpsi{j}{l} = conj(fft2(real(ifft2(tempo.psi{1}{j}{l}))));
%filters{1}.ipsi{j}{l} = conj(fft2(imag(ifft2(tempo.psi{1}{j}{l}))));
%filters{2}.ipsi{j}{l} = conj(fft2(imag(ifft2(tempo.psi{1}{j}{l}))));
%out{s}{t}.ipsi{j} = conj(fft(imag(ifft(in{s}{t}.psi{j}))));
end
end

%%lowpass 
filters{1}.phi = tempo.phi{1};
downsampling=getoptions(options,'downsampling',J-0);
M=N*N*2^(-2*(downsampling));

if 1
AJ=zeros(M,N*N);

dummy=zeros(N);
for n1=1:N
for n2=1:N
dummy=0*dummy;
dummy(n1,n2)=1;
aux=ifft2(fft2(dummy).*tempo.phi{1});
aux=aux(1:2^downsampling:end,1:2^downsampling:end);
AJ(:,N*(n1-1)+n2)=aux(:);
end
end
else
AJ=ones(1,N*N);
end

filters{1}.AJ = AJ;
filters{2}.AJ = AJ;%ones(1,N*N);
