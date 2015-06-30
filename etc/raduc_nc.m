function raduc_nc

% Rescaled Average Degree in the Non-Compact FLRW Causet
% Generates lookup table for CausalSet program
% Written by Will Cunningham

clear;

tauMin = 0.05;
tauMax = 2.00;
stepSize = 0.001;
numSamples = int32((tauMax - tauMin) / stepSize + 1);

k = zeros(numSamples,1);
t = zeros(numSamples,1);

for i = tauMin : stepSize : tauMax
    tau0 = i;
    Z = 8*pi/(sinh(3*tau0)-3*tau0);
    eta = @(x) 2*sinh(1.5*x).^(1/3).*hypergeom([1/6,1/2],7/6,-sinh(1.5*x).^2);
    F = @(x,y) (sinh(1.5*x).^2).*(sinh(1.5*y).^2).*(abs(eta(x)-eta(y)).^3);
    kappa = Z*integral2(F,0,tau0,0,tau0);
    
    index = int32((i - tauMin) / stepSize + 1);
    k(index) = kappa;
    t(index) = tau0;
    
    % fprintf('kappa: %f\n', k(index));
    % fprintf('tau0:  %f\n\n', t(index));
    % drawnow();
end

vals = [k, t]';

fid = fopen('raducNC_table.cset.bin', 'w');
fwrite(fid, vals, 'double');
fclose(fid);

disp('Success');
exit

end
