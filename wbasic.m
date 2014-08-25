function out=wbasic(in, filters, J, L, NN, sizein)

second = 0;

fin = fft2(in);
out=zeros(sizein,1);
r=0;
for j=1:J
    for l=1:L
        M{j}{l}=ifft2(fin.*filters{1}.psi{j}{l});
	ds = 2^(max(0,j-1));
        U{j}{l}=ds * abs(M{j}{l}(1:ds:end,1:ds:end));
        %if second
        %ftmp = fft2(U{j}{l});
        %for jj=j+1:J
        %    for ll=1:L
        %        starget{j}{l}{jj}{ll} = abs(ifft2(ftmp.*filters{1}.psi{jj}{ll}));
        %    end
        %end
        %starget{j}{l}{J+1} = ifft2(ftmp.*filters{1}.phi);
        %else
        starget{j}{l} = U{j}{l};
        %end
	Ntmp = NN/(ds*ds);
	out(r+1:r+Ntmp)=starget{j}{l}(:);r=r+Ntmp;
    end
end
starget{J+1} = ifft2(fin .* filters{1}.phi);
ds = 2^J;
starget{J+1} = ds * starget{J+1}(1:ds:end,1:ds:end);
out(r+1:r+NN*(2^(-2*J))) = starget{J+1}(:);

