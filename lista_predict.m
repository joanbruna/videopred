function out=lista_predict(X, S, E, D, bias, options)

batchsize = size(X, 2);
steps = getoptions(options,'steps',1);

bibi = repmat(bias,1,batchsize);

%fprop
B= E * X;
Z{1} = (abs(B) > bibi).*(B - sign(B).*bibi);
for s=1:steps
[C{s+1}, Z{s+1} ] = lista_fp(Z{s}, S, B, bibi);
end
out = D * Z{steps+1};


