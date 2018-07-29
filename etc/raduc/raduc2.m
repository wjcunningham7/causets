function raduc2

% Rescaled Average Degree in the Compact FLRW Causet
% Version 2
% Generates lookup table for CausalSet program
% Written by Will Cunningham

clear;

%tauMin = 0.05;
%tauMax = 2.00;
%stepSize = 0.001;

%tauMin = 0.05;
%tauMax = 5.00;
%stepSize = 0.05;

tauMin = 7.50;
tauMax = 10.0;
stepSize = 0.1;

numSamples = int32((tauMax - tauMin) / stepSize + 1);

k = zeros(numSamples,1);
t = zeros(numSamples,1);

for i = tauMin: stepSize: tauMax
    tau0 = i;
    r0 = (sinh(1.5*tau0))^(2/3);
    Z = (sinh(3*tau0) - 3*tau0) / 6;
    zeta = @(x) 2*sqrt(x).*hypergeom([1/6,1/2],7/6,-x.^(3));
    F = @(x,y) (abs(zeta(x)-zeta(y)).^3).*(x.^2).*(y.^(7/2))./(sqrt(1+x.^(-3)).*sqrt(1+y.^3));
    kappa = (4*pi/3).*integral2(F,0,r0,0,r0)./Z;
    
    index = int32((i - tauMin) / stepSize + 1);
    %fprintf('index: %d\n', index);
    k(index) = kappa;
    t(index) = tau0;
    
    %fprintf('r0:    %f\n', r0);
    %fprintf('Z:     %f\n', Z);
    fprintf('kappa: %f\n', k(index));
    fprintf('tau0:  %f\n\n', t(index));
end

vals = [k, t]';

fid = fopen('raduc_table.cset.bin', 'w');
fwrite(fid, vals, 'double');
fclose(fid);

disp('Success');
exit

end
