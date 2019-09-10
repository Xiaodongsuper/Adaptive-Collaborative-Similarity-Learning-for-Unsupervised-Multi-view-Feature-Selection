% An out-of-date version, just for reference

%  min  || Ax - y||^2
%  s.t. x>=0, 1'x=1
function [x obj]=SimplexRepresentation_acc(A, y, x0)


NIter = 500;
NStop = 20;

[dim n] = size(A);
%AA = A'*A; 
Ay = A'*y;
if nargin < 3
    x = 1/n*ones(n,1);
else
    x = x0;
end;

x1 = x;
t = 1;
t1 = 0;
r = 0.5;
%obj = zeros(NIter,1);
for iter = 1:NIter
    p = (t1-1)/t;
    s = x + p*(x-x1);
    x1 = x;
    g = A'*(A*s) - Ay; %g = AA*s - Ay;
    ob1 = norm(A*x - y);
    for it = 1:NStop
        z = s - r*g;
        z = EProjSimplex(z);
        ob = norm(A*z - y);
        if ob1 < ob
            r = 0.5*r;
        else
            break;
        end;
    end;
    if it == NStop
        obj(iter) = ob;
        %disp('not');
        break;
    end;
    x = z;
    t1 = t;
    t = (1+sqrt(1+4*t^2))/2;
    
    
    obj(iter) = ob;
end
   
1;
    
