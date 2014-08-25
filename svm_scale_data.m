function [data1, data2]= svm_scale_data(d1, d2)

temp = [d1 d2];

tmin = min(temp,[],2);
tmax = max(temp,[],2);

data1 = ((d1 - repmat(tmin,1,size(d1,2)))./(repmat(.5*(tmax-tmin),1,size(d1,2)))) - 1;
data2 = ((d2 - repmat(tmin,1,size(d2,2)))./(repmat(.5*(tmax-tmin),1,size(d2,2)))) - 1;

