function [paramsout, stateout] = update_params(paramsin, gradin, lr, statein)

stateout=statein;
rho = 0.9;
epsilon = 1e-6;
if 0
%vanilla SGD
for l=1:size(paramsin,2)
paramsout{l} = paramsin{l} - lr * gradin{l};
end

elseif 1

for l=1:size(paramsin,2)
stateout.var{l} = rho * stateout.var{l} - lr * gradin{l};
paramsout{l} = paramsin{l} + stateout.var{l};
end

else
%ADADELTA
%(S, E, bias, D)
for l=1:size(paramsin,2)
tmp2 = gradin{l}.^2;
stateout.gvar{l}= rho * stateout.gvar{l} + (1-rho) * tmp2;
delta = - (sqrt(epsilon+stateout.var{l}) ./ sqrt(epsilon+tmp2)) .* gradin{l};
stateout.var{l} = rho * stateout.var{l} + (1-rho) * (delta.^2);
paramsout{l} = paramsin{l} + delta;
end

end


end


