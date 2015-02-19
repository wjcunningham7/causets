(* ::Package:: *)

(* (C) Will Cunningham 2015 *)
(*  Krioukov Research Group *)
(*  Northeastern University *)

(* This program generates the lookup table used for geodesics in the CasualSet program *)
(* Output is stored in binary format *)

Print[StandardForm["Initializing Constants..."]];

(* Number of Terms in Puiseux Series *)
Nc = 4;

(* Fatness Parameter *)
\[Alpha] = 1;

(* Coarse Step Size *)
cstep = 0.1;

(* Fine Step Size *)
fstep = 0.01;

(* Granularity *)
gran = cstep / fstep;

(* Rescaled Time Interval *)
\[Tau]min = 0;
(* \[Tau]max = 2; *)
\[Tau]max = 0.5;
\[Tau]cells = (\[Tau]max - \[Tau]min) / cstep;
\[Tau]cellsf = \[Tau]cells * gran;

(* X Interval *)
xmin = 1.5 * \[Tau]min;
xmax = 1.5 * \[Tau]max;
xcells = (xmax - xmin) / cstep;
xcellsf = xcells * gran;

(* Lambda Parameter Interval *)
\[Lambda]min = -0.2;
\[Lambda]max = 0.2;
\[Lambda]cells = (\[Lambda]max - \[Lambda]min) / cstep;
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

(* Debug t0 *)
(* tp0 = Table[With[{k = a}, Module[{x0, \[Lambda]0},
	x0 = xi[tm[\[Lambda]val[k]]];
	\[Lambda]0 = \[Lambda]val[k];
	With[{x = x0, \[Lambda] = \[Lambda]0}, t0[[k]][x, \[Lambda]]]
	]], {a, \[Lambda]cells}];

Print@tp0; *)

Print[StandardForm["Generating Table of Puiseux Functions..."]];

(* Table of Approximate Functions *)
t1 = Table[With[{i = a, j = b}, Function[{x, \[Lambda]}, Module[{kern},
	kern[y_, \[Mu]_, m_, n_] = (2/(3*\[Alpha]))*Integrate[N[Normal[Series[h[y, \[Mu]], {y, xval[m], Nc}, {\[Mu], \[Lambda]val[n], Nc}]]], y];
	If[((\[Lambda]val[j] < 0 && g[xval[i], \[Lambda]val[j]] > 0) || \[Lambda]val[j] > 0) && xval[i] != 0, kern[x, \[Lambda], i, j], 0]
	]]], {a, xcells}, {b, \[Lambda]cells}];

(* Debug t1 *)
(* tp1 = Table[With[{i = a, j = b}, Module[{x0, \[Lambda]0},
	x0 = xval[i];
	\[Lambda]0 = \[Lambda]val[j];
	With[{x = x0, \[Lambda] = \[Lambda]0}, t1[[i,j]][x, \[Lambda]]]
	]], {a, xcells}, {b, \[Lambda]cells}];

Print@tp1; *)

Print[StandardForm["Generating Raw Lookup Table..."]];

(* Lookup Table *)
t2 = Table[With[{i = a, j = b, k = c}, Module[{ic, jc, kc},
	ic = Floor[(i - 1) / gran] + 1;
	jc = Floor[(j - 1) / gran] + 1;
	kc = Floor[(k - 1) / gran] + 1;
	If[i >= j, 0,
		If[\[Lambda]valf[k] > 0, 
			With[{x = xvalf[j], \[Lambda] = \[Lambda]valf[k]}, t1[[jc,kc]][x, \[Lambda]]] - With[{x = xvalf[i], \[Lambda] = \[Lambda]valf[k]}, t1[[ic,kc]][x, \[Lambda]]],
			2 * With[{x = xi[tm[\[Lambda]valf[k]]], \[Lambda] = \[Lambda]valf[k]}, t0[[kc]][x, \[Lambda]]] - With[{x = xvalf[j], \[Lambda] = \[Lambda]valf[k]}, t1[[jc,kc]][x, \[Lambda]]] - With[{x = xvalf[i], \[Lambda] = \[Lambda]valf[k]}, t1[[ic,kc]][x, \[Lambda]]]
		]
	]]], {a, xcellsf}, {b, xcellsf}, {c, \[Lambda]cellsf}];

Print[StandardForm["Formatting Lookup Table..."]];

(* Create Points {\[Tau]1, \[Tau]2, \[Omega]12, \[Lambda]} *)
t3 = Table[With[{i = a}, Module[{x0, y0, \[Lambda]0},
	x0 = Floor[(i - 1) / (xcellsf * \[Lambda]cellsf)] + 1;
	y0 = Floor[Mod[(i - 1), xcellsf * \[Lambda]cellsf] / \[Lambda]cellsf] + 1;
	\[Lambda]0 = Floor[Mod[Mod[(i - 1), xcellsf * \[Lambda]cellsf], \[Lambda]cellsf]] + 1;
	{\[Tau]i[xvalf[x0]], \[Tau]i[xvalf[y0]], t2[[x0,y0,\[Lambda]0]], \[Lambda]valf[\[Lambda]0]}
	]], {a, xcellsf * xcellsf * \[Lambda]cellsf}];

Print[StandardForm["Writing Results to File..."]];

(* Write Table to Binary File *)
file = OpenWrite["geodesics_table.cset.bin", BinaryFormat -> True];
BinaryWrite[file, Flatten[t3], "Real64"];
Close[file];

Print[StandardForm["PROGRAM COMPLETED."]];

Exit[];
