function out = wmod_bprop_pinv(in, G, filters, dualfilters)

%out=wmod_bprop(in, G, filters);
%return;

J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);

th=0;%1e-0;

fin=fft2(in);
out=0*in;
for j=1:J
    for l=1:L
        tmp = angle(ifft2(fin.*filters{1}.psi{j}{l}));
        tmp = exp(+i*tmp).* G{j}{l};% .*(abs(G{j}{l}) > th);
        out= out + ifft2(fft2(tmp).*(dualfilters{1}.psi{j}{l}));
    end
end
out = out + ifft2(fft2(G{J+1}).*(dualfilters{1}.phi));
out = real(out);



