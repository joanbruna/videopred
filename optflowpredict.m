function [out,flow]=optflowpredict(im1, im2)

%compute the flow between im1 and im2. 
%predict im3 as the warping by the estimated flow (assumption that flow stays as it is). 

aa(:,:,1)=im1;
aa(:,:,2)=im1;
aa(:,:,3)=im1;
bb(:,:,1)=im2;
bb(:,:,2)=im2;
bb(:,:,3)=im2;

sigma=0.5;
alpha=10;
beta=200;
gamma=5;

flow=mex_LDOF(bb,aa,sigma, alpha, beta, gamma);
flow = permute(flow,[3 1 2]);
out=warp2d(im2, flow);




