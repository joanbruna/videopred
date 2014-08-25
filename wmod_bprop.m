function out=wmod_bprop(in, G, filters)

J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);

out=zeros(size(in));

fin=fft2(in);
for j=1:J
    for l=1:L
        tmp = ifft2(fin.*filters{1}.psi{j}{l});
        tutu = pooling_bprop2d( tmp, G{j}{l}, j, l, filters{1});
       out = out +  tutu;  
        end
end
out = out + ifft2(fft2(G{J+1}).*conj(filters{1}.phi));


        
