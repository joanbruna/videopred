
close all
clear all


N=32;
options.J=3;

filters=prepare_filters_2d(N, options);
J=size(filters{1}.psi,2);
L=size(filters{1}.psi{1},2);

t1=load('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord32.mat');

T=min(400000,size(t1.Xout,2));

data = reshape(t1.dewhitenMatrix * t1.Xout(:,1:T), N, N, T );
verb = 1000;
NN=N*N;
scdim = NN*L*(4-4^(-J+1))/3 + NN*(4^(-J));
out = zeros(scdim, T);

for t=1:T
out(:, t) = wbasic(data(:,:,t), filters, J, L, NN, scdim);
if mod(t, verb)==verb-1
fprintf('done %d \n', t+1)
end
end

[scout, sc_mean, scwhitenMatrix, scdewhitenMatrix] = whiten_jojo(out);

t2.Xout = scout;
clear scout;
t2.mu = sc_mean ; 
clear sc_mean;
t2.whitenMatrix = scwhitenMatrix;
clear scwhitenMatrix;
t2.dewhitenMatrix = scdewhitenMatrix ; 
clear scdewhitenMatrix;

save('/misc/vlgscratch3/LecunGroup/bruna/charles_data_bord32sc.mat','t2', '-v7.3');

