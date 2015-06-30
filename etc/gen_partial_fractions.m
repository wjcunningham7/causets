(* ::Package:: *)

(*(C) Will Cunningham 2015 *)
(* Krioukov Research Group *)
(* Northeastern University *)

(* This package generates the coefficients used for partial fractions
in the geodesic equations implemented in the CausalSet program.
Output is stored in binary format.*)

nkernels = ToExpression[$CommandLine[[Length[$CommandLine]]]];
CloseKernels[];
LaunchKernels[nkernels];

Print[StandardForm["Beginning Generation of Partial Fraction Coefficients."]]; 
Print[StandardForm[""]]; 

(* Constants and Functions *)
nmax = 10;

\[Gamma]1[i_, j_, n_] := (-1)^(3n-j)Binomial[3n, j]Hypergeometric2F1[i-3n, -j, 3n-j+1,-1];
\[Delta]1[i_, j_, n_] := (-1)^(3n-i-j)Binomial[3n-i, j]Hypergeometric2F1[-j, -3n, 3n-i-j+1, -1];

\[Gamma]2a[j_, n_] := (-1)^(3n-j)Gamma[3n+1]/Gamma[j+1];
\[Delta]2a[i_, j_, n_] := (-1)^(3n-i-j)Gamma[3n-i+1]/Gamma[j+1];

\[Gamma]2b[i_, j_, k_, n_] := Hypergeometric2F1Regularized[i-3n, -j, k, -1];
\[Delta]2b[j_, k_, n_] := Hypergeometric2F1Regularized[-j, -3n, k, -1];

(* Matrix Elements for the Four Sub-matrices *)
g1[n_] := Table[\[Gamma]1[i, j, n], {i, 1, 3n}, {j, 0, 3n-1}];

g2val[i_, j_, n_] := If[j == 3n, \[Gamma]1[i, j, n], \[Gamma]2a[j, n]\[Gamma]2b[i, j, 3n-j+1, n]];
g2[n_] := Table[g2val[i, j, n], {i, 1, 3n}, {j, 3n, 6n-1}];

d1val[i_, j_, n_] := Limit[\[Delta]1[x, j, n], x -> i];
d1[n_] := Table[d1val[i, j, n], {i, 1, 3n}, {j, 0, 3n-1}];

d2val[i_, j_, n_] := \[Delta]2a[i, j, n]\[Delta]2b[j, 3n-i-j+1, n];
d2[n_] := Table[d2val[i, j, n], {i, 1, 3n}, {j, 3n, 6n-1}];

(* Combine the two gamma matrices horizontally *)
g12[n_] := Partition[Flatten[Riffle[g1[n], g2[n]]], 6n];
(* Combine the two delta matrices horizontally *)
d12[n_] := Partition[Flatten[Riffle[d1[n], d2[n]]], 6n];
(* Combine the g12 and d12 matrices vertically *)
c[n_] := Partition[Flatten[{g12[n], d12[n]}], 6n];

(* Invert the whole matrix and extract the set of coefficients *)
cinv[n_] := Inverse[c[n]];
vals[n_] := Module[{},
	Print[StandardForm["Calculating Coefficients in Term "], n];
	Print[StandardForm[""]];
	cinv[n][[1]]
	];

(* Whole data table prepared to be written to file *)
data = ParallelTable[N[vals[i]], {i, 1, nmax}];

(* Write to Binary File *)
file = OpenWrite["partial_fraction_coefficients.cset.bin", BinaryFormat -> True];
BinaryWrite[file, Flatten[data], "Real64"];
Close[file];

Print[StandardForm["PROGRAM COMPLETED."]];
CloseKernels[];

Exit[];
