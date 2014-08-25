function [U] = wmod_fprop(in, filters)

finput=fft2(in);
J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);


for j=1:J
    for l=1:L
        M{j}{l}=ifft2(finput.*filters{1}.psi{j}{l});
        U{j}{l}=abs(M{j}{l});
    end
end
U{J+1} = ifft2(finput.*filters{1}.phi);



