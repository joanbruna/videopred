function [gout,ncur] = scatt_grad_step(input, finput, target, filters, dualfilters, second)
%this function does everything. Computes the forward pass until second layer, 
%then backprops difference wrt target to the input, modifies ref accordingly.

lissage = 1;

%forward pass 
J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);

ncur=0;
%fin=fft2(ref);
for j=1:J
    gnorm(j)=0;
    for l=1:L
        M{j}{l}=ifft2(finput.*filters{1}.psi{j}{l});
        U{j}{l}=abs(M{j}{l});
        if second
        ftmp = fft2(U{j}{l});
        for jj=1:j
            for ll=1:L
                dist{jj}{ll} = 0*ftmp;
            end
        end
        for jj=j+1:J
            for ll=1:L
                tmp = abs(ifft2(ftmp.*filters{1}.psi{jj}{ll}));
                if lissage
                dist{jj}{ll} = ifft2(fft2(tmp - target{j}{l}{jj}{ll}).*abs(filters{1}.phi).^2);
                else
                dist{jj}{ll} = tmp - target{j}{l}{jj}{ll};
                end
            end
        end
        tmp = ifft2(ftmp.*filters{1}.phi);
        dist{J+1} = tmp - target{j}{l}{J+1};
        grad{j}{l} = wmod_bprop_pinv(U{j}{l}, dist, filters, dualfilters);
        else
            if lissage
        grad{j}{l} = ifft2(fft2(U{j}{l} - target{j}{l}).*abs(filters{1}.phi).^2);
            else
        grad{j}{l} = U{j}{l} - target{j}{l};
	ncur = ncur + sum(grad{j}{l}(:).^2);
            end
        end
        %gnorm(j) = gnorm(j) + norm(grad{j}{l}(:))^2;
        clear dist;
    end
    %fprintf('gnorm scale %d is %f \n', j, gnorm(j))
end
U{J+1} = ifft2(finput.*filters{1}.phi);
grad{J+1} = U{J+1} - target{J+1};
ncur = ncur + sum(grad{J+1}(:).^2);

    %fprintf('gnorm scale %d is %f \n', J+1, norm(grad{J+1}(:))^2)



gout = wmod_bprop_pinv(input, grad, filters, dualfilters);

%out = input - lambda * gout;

 



