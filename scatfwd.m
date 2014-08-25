function starget = scatfwd(target, filters, second)

J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);


ftarget = fft2(target);
for j=1:J
    for l=1:L
        M{j}{l}=ifft2(ftarget.*filters{1}.psi{j}{l});
        U{j}{l}=abs(M{j}{l});
        if second
        ftmp = fft2(U{j}{l});
        for jj=j+1:J
            for ll=1:L
                starget{j}{l}{jj}{ll} = abs(ifft2(ftmp.*filters{1}.psi{jj}{ll}));
            end
        end
        starget{j}{l}{J+1} = ifft2(ftmp.*filters{1}.phi);
        else
        starget{j}{l} = U{j}{l};
        end
    end
end
starget{J+1} = ifft2(ftarget .* filters{1}.phi);


