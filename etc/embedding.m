function embedding

clear;

c = 1;
a = 2;
alpha = (c.*a.^2).^(1./3);

bins = linspace(0,20,100)';

d = @(x) (a.*alpha).^2 .* x ./ (alpha.^3 + x.^3);
ep = @(x) (1 + d(x)).^(1./2);
em = @(x) (1 - d(x)).^(1./2);

fp = zeros(length(bins),1);
fm = zeros(length(bins),1);

for i = 1 : length(bins)
    fp(i) = integral(ep,0,bins(i));
    fm(i) = integral(em,0,bins(i));
end

scatter(bins, fp);
axis([0 20 0 25]);

end