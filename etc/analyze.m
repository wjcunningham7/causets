function analyze

clear;

N = 10240;

figure;
k = load('average_k.cset.deg.ref');
nb = 25;
k_bins = linspace(min(k), max(k) + 1, nb)';
pk = histc(k, k_bins);
pk = pk ./ sum(pk);
bar(k_bins, pk);
axis([0 max(k)+1 0 0.15]);

figure;
Niso = load('average_N_iso.cset.cmp.ref');
Niso_bins = linspace(min(Niso), max(Niso) + 1, 2*nb)';
pNiso = histc(Niso, Niso_bins);
pNiso = pNiso ./ sum(pNiso);
bar(Niso_bins, pNiso);
axis([0 500 0 0.5]);

figure;
Ncc = load('average_N_cc.cset.cmp.ref');
Ncc_bins = linspace(min(Ncc), max(Ncc) + 1, 2*nb)';
pNcc = histc(Ncc, Ncc_bins);
pNcc = pNcc ./ sum(pNcc);
bar(Ncc_bins, pNcc);
axis([0 50 0 0.7]);

figure;
Ngcc = load('average_N_gcc.cset.cmp.ref');
Ngcc_bins = linspace(min(Ngcc), max(Ngcc) + 1, nb)';
pNgcc = histc(Ngcc, Ngcc_bins);
pNgcc = pNgcc ./ sum(pNgcc);
bar(Ngcc_bins, pNgcc);
axis([8500 N 0 0.7]);

end