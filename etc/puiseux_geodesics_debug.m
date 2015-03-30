(* ::Package:: *)

(* (C) Will Cunningham 2015 *)
(*  Krioukov Research Group *)
(*  Northeastern University *)

(* This program tests individual lookup table values *)
(* This should be used for debugging the table *)

SetOptions[$Output, FormatType->OutputForm];
nkernels = ToExpression[$CommandLine[[Length[$CommandLine]]]];
CloseKernels[];
LaunchKernels[nkernels];

Print[StandardForm["Initializing Constants..."]];

(* Number of Terms in Puiseux Series *)
Nc = 6;

(* Fatness Parameter *)
\[Alpha] = 1;

(* Coarse Step Size *)
cstep = 0.1;

(* Fine Step Size *)
fstep = 0.05;

(* Granularity *)
gran = cstep / fstep;

(* Rescaled Time Interval *)
\[Tau]min = 0;
(* \[Tau]max = 2; *)
\[Tau]max = 0.1;
\[Tau]cells = (\[Tau]max - \[Tau]min) / cstep + 1;
\[Tau]cellsf = \[Tau]cells * gran;

(* X Interval *)
xmin = 1.5 * \[Tau]min;
xmax = 1.5 * \[Tau]max;
xcells = (xmax - xmin) / cstep + 1;
xcellsf = xcells * gran;

(* Lambda Parameter Interval *)
\[Lambda]min = -0.5;
\[Lambda]max = 0;
\[Lambda]cells = (\[Lambda]max - \[Lambda]min) / cstep + 1;
\[Lambda]cellsf = \[Lambda]cells * gran;

(* Spatial Kernel Function *)
g[x_, \[Lambda]_] := Sinh[x]^(4/3) + \[Lambda] * Sinh[x]^(8/3);
h[x_, \[Lambda]_] := (g[x, \[Lambda]])^(-1/2);

(* Maximum Time, for \[Lambda] < 0 *)
tm[\[Lambda]_] := If[\[Lambda] != 0, (2/3) * ArcSinh[Abs[\[Lambda]]^(-3/4)], 0, 0];

(* Helper Functions *)
xi[\[Tau]_] := 1.5 * \[Tau];
\[Tau]i[x_] := x / 1.5;
xval[i_] := (i - 1) * cstep + xmin;
xvalf[i_] := (i - 1) * fstep + xmin;

\[Lambda]val[i_] := (i - 1) * cstep + \[Lambda]min;
\[Lambda]valf[i_] := (i - 1) * fstep + \[Lambda]min;

Print[StandardForm["Generating Table of Maximum Time Functions..."]];

(* Table of Max Time Functions *)
t0 = Table[With[{i = a}, Function[{x, \[Lambda]}, Module[{kern},
	kern[y_, \[Mu]_, j_] = (2/(3*\[Alpha]))*Integrate[N[Normal[Series[h[y, \[Mu]], {y, xi[tm[\[Lambda]val[j]]], Nc}, {\[Mu], \[Lambda]val[j], Nc}]]], y];
	If[\[Lambda]val[i] < 0 && g[xi[tm[\[Lambda]val[i]]], \[Lambda]val[i]] > 0, kern[x, \[Lambda], i], 0]
	]]], {a, \[Lambda]cells}];

Print[StandardForm["Generating Table of Puiseux Functions..."]];

(* Table of Approximate Functions *)
t1 = Table[With[{i = a, j = b}, Function[{x, \[Lambda]}, Module[{kern},
	kern[y_, \[Mu]_, m_, n_] = (2/(3*\[Alpha]))*Integrate[N[Normal[Series[h[y, \[Mu]], {y, xval[m], Nc}, {\[Mu], \[Lambda]val[n], Nc}]]], y];
	If[((\[Lambda]val[j] < 0 && g[xval[i], \[Lambda]val[j]] > 0) || \[Lambda]val[j] > 0) && xval[i] != 0, kern[x, \[Lambda], i, j], 0]
	]]], {a, xcells}, {b, \[Lambda]cells}];

i = 1;
j = 2;
k = 1;
ic = Floor[(i - 1) / gran] + 1;
jc = Floor[(j - 1) / gran] + 1;
kc = Floor[(k - 1) / gran] + 1;

Print["tau1: ", \[Tau]i[xvalf[i]]];
Print["tau2: ", \[Tau]i[xvalf[j]]];
Print["lambda: ", \[Lambda]valf[k]];

Print@If[i >= j, 0,
	If[\[Lambda]valf[k] > 0, 
		With[{x = xvalf[j], \[Lambda] = \[Lambda]valf[k]}, t1[[jc,kc]][x, \[Lambda]]] - With[{x = xvalf[i], \[Lambda] = \[Lambda]valf[k]}, t1[[ic,kc]][x, \[Lambda]]],
		2 * With[{x = xi[tm[\[Lambda]valf[k]]], \[Lambda] = \[Lambda]valf[k]}, t0[[kc]][x, \[Lambda]]] - With[{x = xvalf[j], \[Lambda] = \[Lambda]valf[k]}, t1[[jc,kc]][x, \[Lambda]]] - With[{x = xvalf[i], \[Lambda] = \[Lambda]valf[k]}, t1[[ic,kc]][x, \[Lambda]]]
	]
];

Print[StandardForm["Generating Exact Solution..."]];

\[Omega][\[Tau]1_, \[Tau]2_, \[Lambda]_] := With[{\[Mu] = \[Lambda], max = tm[\[Lambda]]}, 
	If[\[Mu] > 0, 
		NIntegrate[h[\[Tau], \[Mu]], {\[Tau], \[Tau]1, \[Tau]2}],
		NIntegrate[h[\[Tau], \[Mu]], {\[Tau], \[Tau]1, max}] + NIntegrate[h[\[Tau], \[Mu]], {\[Tau], \[Tau]2, max}]
	]
];

Print[With[{\[Sigma]1 = \[Tau]i[xvalf[i]], \[Sigma]2 = \[Tau]i[xvalf[j]], \[Nu] = \[Lambda]valf[k]}, \[Omega][\[Sigma]1, \[Sigma]2, \[Nu]]]];

Print[StandardForm["PROGRAM COMPLETED."]];
CloseKernels[];

Exit[];