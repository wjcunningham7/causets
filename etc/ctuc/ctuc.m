function ctuc

% Conformal Time in the Universe Causet
% Generates lookup table for CausalSet program
% Written by Will Cunningham

clear;

tauMin = 0.05;
tauMax = 10.0;
stepSize = 0.05;

numSamples = int32((tauMax - tauMin) / stepSize + 1);

f = zeros(numSamples,1);
t = zeros(numSamples,1);

for i = tauMin: stepSize: tauMax
    tau = i;
    s = sech(1.5*i)^2;
    h = hypergeom([1/3,5/6],[4/3],s);

    index = int32((i - tauMin) / stepSize + 1);
    f(index) = h*s^(1/3);
    t(index) = tau;
end

vals = [f, t]';

fid = fopen('ctuc_table.cset.bin','w');
fwrite(fid, vals, 'double');
fclose(fid);

disp('Success');
exit

end
