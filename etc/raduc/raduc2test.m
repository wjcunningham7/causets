function raduc2test

% Rescaled Average Degree in the Universe Causet
% Version 2 (Test)
% Generates single value from lookup table
% Written by Will Cunningham

clear;

tau0 = 6.0;
r0 = (sinh(1.5*tau0))^(2/3);
Z = (sinh(3*tau0) - 3*tau0) / 6;
zeta = @(x) 2*sqrt(x).*hypergeom([1/6,1/2],7/6,-x.^(3));
F = @(x,y) (abs(zeta(x)-zeta(y)).^3).*(x.^2).*(y.^(7/2))./(sqrt(1+x.^(-3)).*sqrt(1+y.^3));
kappa = (4*pi/3).*integral2(F,0,r0,0,r0)./Z;
disp(kappa);
disp('Success');
exit

end
