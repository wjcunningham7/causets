% Concatenates binary ctuc_table files
% Written by Will Cunningham

clear;

% Open target files and read entries

f1 = fopen('ctuc_table_high_res.cset.bin');
d1 = fread(f1, [2, Inf], 'double')';

f2 = fopen('ctuc_table_med_res.cset.bin');
d2 = fread(f2, [2, Inf], 'double')';

fclose('all');

% Concatenate arrays

d = vertcat(d1, d2);

% Write to single output file

fid = fopen('ctuc_table.cset.bin','w');
fwrite(fid, d', 'double');
fclose(fid);

% Check for success

f4 = fopen('ctuc_table.cset.bin');
d4 = fread(f4, [2, Inf], 'double')';
fclose(f4);

disp('Success');
exit
