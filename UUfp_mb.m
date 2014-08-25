function [Xout,scratch] = UUfp_mb(Xin, V, W, Wd, k)


[P, bs] = size(V);
[N, bs] = size(Xin);

scratch.Y = W*Xin ; 
scratch.Z = scratch.Y.* replicate(V,k);
Xout = Wd * scratch.Z ; 


