function dP = phasebp(dPin, Pha, Mod, k)

[M, bs]=size(Pha);
[P, bs]=size(Mod);

dP = (dPin - Pha .* reshape(repmat(sum(reshape( dPin.*Pha, k, P*bs)),[k 1]),M, bs))./replicate(Mod,k);


