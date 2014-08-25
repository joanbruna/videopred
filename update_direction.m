function Gout=update_direction(Gin, current, targ, filters, rnorm)

[Ucurr] = wmod_fprop(current, filters);
%path 
J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);
npool=0;
gvin=[];
gvout=[];
for j=1:J
    for l=1:L
        Gout{j}{l} = targ{j}{l} - Ucurr{j}{l};
        gvout=[gvout; Gout{j}{l}(:)];
        gvin=[gvin; Gin{j}{l}(:)];
        npool = npool + norm(Gout{j}{l}(:))^2;
    end
end
Gout{J+1} = targ{J+1} - Ucurr{J+1};
gvout=[gvout; Gout{J+1}(:)];
gvin=[gvin; Gin{J+1}(:)];

%sum(gvin.*gvout)/(norm(gvin)*norm(gvout));



npool = npool + norm(Gout{J+1}(:))^2;
npool = sqrt(npool);

if rnorm > 0
factor=rnorm/npool;
for j=1:J
    for l=1:L
        Gout{j}{l} = factor * Gout{j}{l};
        end
        end
        Gout{J+1} = Gout{J+1} * factor;
end

