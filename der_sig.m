function [outputArg1] = der_sig(x)
outputArg1 =(1/(1+exp(-x)))*(1-1/(1+exp(-x)));
end  % dervitave sig=sig(x)(1-sig(x))