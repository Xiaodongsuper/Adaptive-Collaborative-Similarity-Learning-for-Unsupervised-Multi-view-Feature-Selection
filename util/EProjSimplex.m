% An out-of-date version, just for reference

function [x ft] = EProjSimplex(v)

%
% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%

ft=1;
n = length(v);

v0 = v-mean(v) + 1/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = lambda_m - v0;
        posidx = v1>0;
        npos = sum(posidx);
        g = npos/n - 1;
        f = sum(v1(posidx))/n-lambda_m;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(-v1,0);
            break;
        end;
    end;
    x = max(-v1,0);

else
    x = v0;
end;
