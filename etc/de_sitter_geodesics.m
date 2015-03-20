(* ::Package:: *)

(* (C) Will Cunningham 2015 *)
(*  Krioukov Research Group *)
(*  Northeastern University *)

(* This program generates the lookup table used for geodesics in de Sitter spacetime in the CausalSet program *)
(* Output is stored in binary format *)

nkernels = ToExpression[$CommandLine[[Length[$CommandLine]]]];
CloseKernels[];
LaunchKernels[nkernels];

Print[StandardForm["Initializing Constants..."]];

(* de Sitter Pseudoradius *)
a = 2.0;

(* Step Size *)
step = 0.01;

(* Rescaled Time Interval *)
\[Tau]min = 0.0;
\[Tau]max = 2.0
\[Tau]cells = (\[Tau]max - \[Tau]min) / step;

(* Lambda Parameter Interval *)
\[Lambda]min = -1.0;
\[Lambda]max = 1.0;
\[Lambda]cells = (\[Lambda]max - \[Lambda]min) / step;

(* Spatial Kernel Function *)
f[\[Tau]_, \[Lambda]_] := ArcTan[Sqrt[2]*Sinh[\[Tau]]*(\[Lambda]*Cosh[2*\[Tau]]+\[Lambda]+2)^(-0.5)];

(* Maximum Time, for \[Lambda] < 0 *)
tm[\[Lambda]_, a_] := If[\[Lambda] != 0, ArcCosh[Abs[\[Lambda]]^(-0.5) / a];

(* Helper Functions *)
\[Tau]val[i_] := (i - 1) * step + \[Tau]min;
\[Lambda]val[i_] := (i - 1) * step + \[Lambda]min;

Print[StandardForm["Generating Raw Lookup Table..."]];

(* Lookup Table *)
t0 = Table[With[{i = a, j = b, k = c},
	If[i >= j, 0,
		If[\[Lambda]val[k] > 0,
			f[\[Tau]val[j], \[Lambda]val[k]] - f[\[Tau]val[i], \[Lambda]val[i]],
			2.0 * f[tm[\[Lambda]val[k], a], \[Lambda]val[k]] - f[\[Tau]val[j], \[Lambda]val[k]] - f[\[Tau]val[i], \[Lambda]val[i]]
		]
	]], {a, \[Tau]cells}, {b, \[Tau]cells}, {c, \[Lambda]cells}];

Print[StandardForm["Formatting Lookup Table..."]];

(* Generate Points (\[Tau]1, \[Tau]2, \[Omega]12, \[Lambda]} *)
t1 = Table[With[{i = a}, Module[{x0, y0, \[Lambda]0},
	x0 = Floor[(i - 1) / (\[Tau]cells* \[Lambda]cells)] + 1;
	y0 = Floor[Mod[(i - 1), \[Tau]cells * \[Lambda]cells] / \[Lambda]cells] + 1;
	\[Lambda]0 = Floor[Mod[Mod[(i - 1), \[Tau]cells * \[Lambda]cells], \[Lambda]cells]] + 1;
	{\[Tau]val[x0], \[Tau]val[y0], t0[[x0,y0,\[Lambda]0]], \[Lambda]val[\[Lambda]0]}
	]], {a, \[Tau]cells * \[Tau]cells * \[Lambda]cells}];

Print[StandardForm["Writing Results to File..."]];

(* Write Table to Binary File *)
file = OpenWrite["ds_geodesics_table.cset.bin", BinaryFormat -> True];
BinaryWrite[file, Flatten[t1], "Real64"];
Close[file];

Print[StandardForm["PROGRAM COMPLETED."]];
CloseKernels[];

Exit[];
