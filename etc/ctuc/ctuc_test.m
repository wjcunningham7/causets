function ctuc_test

% Conformal Time in the Universe Causet
% Version 2 (Test)
% Generates single value from lookup table
% Written by Will Cunningham

clear;
    
tau = 0;
s = sech(1.5*tau)^2;
h = hypergeom([1/3,5/6],[4/3],s);

f = h*s^(1/3);
t = tau;

disp('f(tau):');
disp(f);
disp('Success');
exit

end
