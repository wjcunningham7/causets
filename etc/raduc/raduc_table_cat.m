% Concatenates binary raduc_table files
% Written by Will Cunningham

clear;

% Open target files and read entries

f1 = fopen('raduc_table_med_res.cset.bin');
d1 = fread(f1, [2, Inf], 'double')';

f2 = fopen('raduc_table_low_res.cset.bin');
d2 = fread(f2, [2, Inf], 'double')';

f3 = fopen('raduc_table_low_res_2.cset.bin');
d3 = fread(f3, [2, Inf], 'double')';

fclose('all');

% Remove duplicate entries

h1 = d1;
h2 = d2(2:size(d2,1),:);
h3 = d3(2:size(d3,1),:);

% Concatenate arrays

d = vertcat(h1, h2, h3);

% Write to single output file

fid = fopen('raduc_table.cset.bin','w');
fwrite(fid, d', 'double');
fclose(fid);

% Check for success

f4 = fopen('raduc_table.cset.bin');
d4 = fread(f4, [2, Inf], 'double')';
fclose(f4);