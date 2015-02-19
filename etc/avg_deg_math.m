(* ::Package:: *)

Needs["ToMatlab`", "./ToMatlab.m"];

(* Constants and Functions *)

Print[StandardForm["Avg_Deg_Math Output..."]];
Print[StandardForm[""]];

h1[\[Theta]_] := R0 - Abs[\[Theta]];
h2[\[Theta]_] := R0 + Abs[\[Theta]];
h3 = \[Tau]0 - 2*R0;
w1 = R0 - \[Tau]0;
w2 = \[Tau]0 - R0;
f[\[Theta]_, K_] := Sin[Sqrt[K]*\[Theta]]^2/K;
g[\[Tau]_] := Sinh[3*(\[Tau]/2)]^2;
\[Rho]\[Tau][\[Tau]_] := 6*(Sinh[3*(\[Tau]/2)]^2/(Sinh[3*\[Tau]0] - 3*\[Tau]0));
\[Rho]\[Theta][\[Theta]_] := 3*(\[Theta]^2/R0^3);

(* Future Light Cone at Point (\[Tau]\[Prime], \[Theta]\[Prime]) *)

(* Case I: Regions A, B, C, and D *)
ABCD1\[Theta][\[Tau]\[DoublePrime]_, \[Tau]\[Prime]_, \[Theta]\[Prime]_] = Limit[Integrate[f[\[Theta]\[DoublePrime], K], {\[Theta]\[DoublePrime], \[Theta]\[Prime] - (\[Tau]\[DoublePrime] - \[Tau]\[Prime]), \[Theta]\[Prime] + (\[Tau]\[DoublePrime] - \[Tau]\[Prime])}], K -> 0];
ABCD1\[Tau][\[Tau]\[Prime]_, \[Theta]\[Prime]_] = Integrate[g[\[Tau]\[DoublePrime]]*ABCD1\[Theta][\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Theta]\[Prime]], {\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Tau]\[Prime] + h1[\[Theta]\[Prime]]}];
ABCD1[\[Tau]\[Prime]_, \[Theta]\[Prime]_] = (4*Pi*a*\[Alpha]^3)*ABCD1\[Tau][\[Tau]\[Prime], \[Theta]\[Prime]];
Print[StandardForm["ABCD1:"]];
Print[StandardForm[ToString[ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]], CharacterEncoding -> "ASCII"]]];
Print[StandardForm[""]];

(* Case I: Region E *)
E1\[Theta][\[Tau]\[DoublePrime]_, \[Tau]\[Prime]_, \[Theta]\[Prime]_] = Limit[Integrate[f[\[Theta]\[DoublePrime], K], {\[Theta]\[DoublePrime], \[Theta]\[Prime] - (\[Tau]\[DoublePrime] - \[Tau]\[Prime]), \[Theta]\[Prime] + (\[Tau]\[DoublePrime] - \[Tau]\[Prime])}], K -> 0];
E1\[Tau][\[Tau]\[Prime]_, \[Theta]\[Prime]_] = Integrate[g[\[Tau]\[DoublePrime]]*E1\[Theta][\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Theta]\[Prime]], {\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Tau]0}];
E1[\[Tau]\[Prime]_, \[Theta]\[Prime]_] = (4*Pi*a*\[Alpha]^3)*E1\[Tau][\[Tau]\[Prime], \[Theta]\[Prime]];
Print[StandardForm["E1:"]];
Print[StandardForm[ToString[E1[\[Tau]\[Prime], \[Theta]\[Prime]], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case II: Region A *)
A2\[Theta][\[Tau]\[DoublePrime]_, \[Tau]\[Prime]_, \[Theta]\[Prime]_] = Limit[Integrate[f[\[Theta]\[DoublePrime], K], {\[Theta]\[DoublePrime], -R0, \[Theta]\[Prime] + (\[Tau]\[DoublePrime] - \[Tau]\[Prime])}], K -> 0];
A2\[Tau][\[Tau]\[Prime]_, \[Theta]\[Prime]_] = Integrate[g[\[Tau]\[DoublePrime]]*A2\[Theta][\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Theta]\[Prime]], {\[Tau]\[DoublePrime], \[Tau]\[Prime] + h1[\[Theta]\[Prime]], \[Tau]\[Prime] + (h2[\[Theta]\[Prime]] - h1[\[Theta]\[Prime]])}];
A2[\[Tau]\[Prime]_, \[Theta]\[Prime]_] = (4*Pi*a*\[Alpha]^3)*A2\[Tau][\[Tau]\[Prime], \[Theta]\[Prime]];
Print[StandardForm["A2:"]];
Print[StandardForm[ToString[A2[\[Tau]\[Prime], \[Theta]\[Prime]], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case II: Region B *)
B2\[Theta][\[Tau]\[DoublePrime]_, \[Tau]\[Prime]_, \[Theta]\[Prime]_] = Limit[Integrate[f[\[Theta]\[DoublePrime], K], {\[Theta]\[DoublePrime], \[Theta]\[Prime] - (\[Tau]\[DoublePrime] - \[Tau]\[Prime]), R0}], K -> 0];
B2\[Tau][\[Tau]\[Prime]_, \[Theta]\[Prime]_] = Integrate[g[\[Tau]\[DoublePrime]]*B2\[Theta][\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Theta]\[Prime]], {\[Tau]\[DoublePrime], \[Tau]\[Prime] + h1[\[Theta]\[Prime]], \[Tau]\[Prime] + (h2[\[Theta]\[Prime]] - h1[\[Theta]\[Prime]])}];
B2[\[Tau]\[Prime]_, \[Theta]\[Prime]_] = (4*Pi*a*\[Alpha]^3)*B2\[Tau][\[Tau]\[Prime], \[Theta]\[Prime]];
Print[StandardForm["B2:"]];
Print[StandardForm[ToString[B2[\[Tau]\[Prime], \[Theta]\[Prime]], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case II: Region C *)
C2\[Theta][\[Tau]\[DoublePrime]_, \[Tau]\[Prime]_, \[Theta]\[Prime]_] = Limit[Integrate[f[\[Theta]\[DoublePrime], K], {\[Theta]\[DoublePrime], -R0, \[Theta]\[Prime] + (\[Tau]\[DoublePrime] - \[Tau]\[Prime])}], K -> 0];
C2\[Tau][\[Tau]\[Prime]_, \[Theta]\[Prime]_] = Integrate[g[\[Tau]\[DoublePrime]]*C2\[Theta][\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Theta]\[Prime]], {\[Tau]\[DoublePrime], \[Tau]\[Prime] + h1[\[Theta]\[Prime]], \[Tau]0}];
C2[\[Tau]\[Prime]_, \[Theta]\[Prime]_] = (4*Pi*a*\[Alpha]^3)*C2\[Tau][\[Tau]\[Prime], \[Theta]\[Prime]];
Print[StandardForm["C2:"]];
Print[StandardForm[ToString[C2[\[Tau]\[Prime], \[Theta]\[Prime]], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case II: Region D *)
D2\[Theta][\[Tau]\[DoublePrime]_, \[Tau]\[Prime]_, \[Theta]\[Prime]_] = Limit[Integrate[f[\[Theta]\[DoublePrime], K], {\[Theta]\[DoublePrime], \[Theta]\[Prime] - (\[Tau]\[DoublePrime] - \[Tau]\[Prime]), R0}], K -> 0];
D2\[Tau][\[Tau]\[Prime]_, \[Theta]\[Prime]_] = Integrate[g[\[Tau]\[DoublePrime]]*D2\[Theta][\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Theta]\[Prime]], {\[Tau]\[DoublePrime], \[Tau]\[Prime] + h1[\[Theta]\[Prime]], \[Tau]0}];
D2[\[Tau]\[Prime]_, \[Theta]\[Prime]_] = (4*Pi*a*\[Alpha]^3)*D2\[Tau][\[Tau]\[Prime], \[Theta]\[Prime]];
Print[StandardForm["D2:"]];
Print[StandardForm[ToString[D2[\[Tau]\[Prime], \[Theta]\[Prime]], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case III: Regions A and B *)
AB3\[Theta][\[Tau]\[DoublePrime]_, \[Tau]\[Prime]_, \[Theta]\[Prime]_] = Limit[Integrate[f[\[Theta]\[DoublePrime], K], {\[Theta]\[DoublePrime], -R0, R0}], K -> 0];
AB3\[Tau][\[Tau]\[Prime]_, \[Theta]\[Prime]_] = Integrate[g[\[Tau]\[DoublePrime]]*AB3\[Theta][\[Tau]\[DoublePrime], \[Tau]\[Prime], \[Theta]\[Prime]], {\[Tau]\[DoublePrime], \[Tau]\[Prime] + h2[\[Theta]\[Prime]], \[Tau]0}];
AB3[\[Tau]\[Prime]_, \[Theta]\[Prime]_] = (4*Pi*a*\[Alpha]^3)*AB3\[Tau][\[Tau]\[Prime], \[Theta]\[Prime]];
Print[StandardForm["AB3:"]];
Print[StandardForm[ToString[AB3[\[Tau]\[Prime], \[Theta]\[Prime]], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Averaged Future Light Cone *)

(* Case \[Alpha] : \[Tau]0 > 2R0 *)

(* Region A *)
\[Alpha]A1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + A2[\[Tau]\[Prime], \[Theta]\[Prime]] + AB3[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], -R0, 0}, Assumptions -> \[Tau]0 > 0 && R0 > 0];
\[Alpha]A1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]A1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, h3}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0];
\[Alpha]A2\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + A2[\[Tau]\[Prime], \[Theta]\[Prime]] + AB3[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], -R0 + (\[Tau]\[Prime] - h3), 0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0 && \[Tau]\[Prime] - h3 > 0 && \[Tau]\[Prime] - h3 < R0];
\[Alpha]A2 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]A2\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], h3, h3 + R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0 && h3 > 0];
\[Alpha]A = FullSimplify[\[Alpha]A1 + \[Alpha]A2, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0];
Print[StandardForm["\[Alpha]A:"]];
Print[StandardForm[ToString[\[Alpha]A, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region B *)
\[Alpha]B1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + B2[\[Tau]\[Prime], \[Theta]\[Prime]] + AB3[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], 0, R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0];
\[Alpha]B1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]B1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, h3}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0];
\[Alpha]B2\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + B2[\[Tau]\[Prime], \[Theta]\[Prime]] + AB3[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], 0, R0 - (\[Tau]\[Prime] - h3)}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0 && \[Tau]\[Prime] - h3 > 0 && \[Tau]\[Prime] - h3 < R0];
\[Alpha]B2 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]B2\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], h3, h3 + R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0 && h3 > 0];
\[Alpha]B = FullSimplify[\[Alpha]B1 + \[Alpha]B2, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0];
Print[StandardForm["\[Alpha]B:"]];
Print[StandardForm[ToString[\[Alpha]B, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region C *)
\[Alpha]C1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + C2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], -R0, -R0 + (\[Tau]\[Prime] - h3)}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && \[Tau]\[Prime] - h3 > 0 && \[Tau]\[Prime] - h3 < R0 && \[Tau]0 > 2*R0];
\[Alpha]C1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]C1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], h3, h3 + R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0];
\[Alpha]C2\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + C2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], -R0, (h3 + R0) - \[Tau]\[Prime]}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && h3 + R0 < \[Tau]\[Prime] && \[Tau]0 > \[Tau]\[Prime] && \[Tau]0 > 2*R0];
\[Alpha]C2 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]C2\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], h3 + R0, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && h3 + R0 < \[Tau]0 && \[Tau]0 > 2*R0];
\[Alpha]C = FullSimplify[\[Alpha]C1 + \[Alpha]C2, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0];
Print[StandardForm["\[Alpha]C:"]];
Print[StandardForm[ToString[\[Alpha]C, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region D *)
\[Alpha]D1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + D2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], R0 - (\[Tau]\[Prime] - h3), R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && \[Tau]\[Prime] - h3 > 0 && \[Tau]\[Prime] - h3 < R0 && \[Tau]0 > 2*R0];
\[Alpha]D1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]D1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], h3, h3 + R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0];
\[Alpha]D2\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + D2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], \[Tau]\[Prime] - (h3 + R0), R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && h3 + R0 < \[Tau]\[Prime] && \[Tau]0 > \[Tau]\[Prime] && \[Tau]0 > 2*R0];
\[Alpha]D2 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]D2\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], h3 + R0, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && h3 + R0 < \[Tau]0 && \[Tau]0 > 2*R0];
\[Alpha]D = FullSimplify[\[Alpha]D1 + \[Alpha]D2, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0];
Print[StandardForm["\[Alpha]D:"]];
Print[StandardForm[ToString[\[Alpha]D, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region E *)
\[Alpha]E1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*E1[\[Tau]\[Prime], \[Theta]\[Prime]], {\[Theta]\[Prime], (h3 + R0) - \[Tau]\[Prime], \[Tau]\[Prime] - (h3 + R0)}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && h3 + R0 < \[Tau]\[Prime] && \[Tau]0 > 2*R0];
\[Alpha]E1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Alpha]E1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], h3 + R0, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && h3 > 0 && \[Tau]0 > 2*R0];
\[Alpha]E = FullSimplify[\[Alpha]E1, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0];
Print[StandardForm["\[Alpha]E:"]];
Print[StandardForm[ToString[\[Alpha]E, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case \[Beta] : R0 < \[Tau]0 < 2R0 *)

(* Region A *)
\[Beta]A1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + A2[\[Tau]\[Prime], \[Theta]\[Prime]] + AB3[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], \[Tau]\[Prime] - w2, 0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]\[Prime] - w2 < 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]A1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Beta]A1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, w2}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]A = FullSimplify[\[Beta]A1, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
Print[StandardForm["\[Beta]A:"]];
Print[StandardForm[ToString[\[Beta]A, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region B *)
\[Beta]B1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + B2[\[Tau]\[Prime], \[Theta]\[Prime]] + AB3[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], 0, w2 - \[Tau]\[Prime]}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && w2 - \[Tau]\[Prime] > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]B1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Beta]B1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, w2}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]B = FullSimplify[\[Beta]B1, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
Print[StandardForm["\[Beta]B:"]];
Print[StandardForm[ToString[\[Beta]B, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region C *)
\[Beta]C1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + C2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], -R0, \[Tau]\[Prime] - w2}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]\[Prime] - w2 > -R0 && \[Tau]\[Prime] - w2 < 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]C1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Beta]C1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, w2}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]C2\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + C2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], -R0, w2 - \[Tau]\[Prime]}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && w2 - \[Tau]\[Prime] > -R0 && w2 - \[Tau]\[Prime] < 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]C2 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Beta]C2\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], w2, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && w2 < \[Tau]0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]C = FullSimplify[\[Beta]C1 + \[Beta]C2, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
Print[StandardForm["\[Beta]C:"]];
Print[StandardForm[ToString[\[Beta]C, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region D *)
\[Beta]D1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + D2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], w2 - \[Tau]\[Prime], R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && w2 - \[Tau]\[Prime] < R0 && w2 - \[Tau]\[Prime] > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]D1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Beta]D1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, w2}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]D2\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + D2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], \[Tau]\[Prime] - w2, R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]\[Prime] - w2 < R0 && \[Tau]\[Prime] - w2 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]D2 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Beta]D2\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], w2, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && w2 < \[Tau]0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]D = FullSimplify[\[Beta]D1 + \[Beta]D2, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
Print[StandardForm["\[Beta]D:"]];
Print[StandardForm[ToString[\[Beta]D, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region E *)
\[Beta]E1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*E1[\[Tau]\[Prime], \[Theta]\[Prime]], {\[Theta]\[Prime], w2 - \[Tau]\[Prime], \[Tau]\[Prime] - w2}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && \[Tau]\[Prime] - w2 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]E1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Beta]E1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], w2, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w2 > 0 && w2 < \[Tau]0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
\[Beta]E = FullSimplify[\[Beta]E1, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0];
Print[StandardForm["\[Beta]E:"]];
Print[StandardForm[ToString[\[Beta]E, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case \[Gamma] : \[Tau]0 < R0 *)

(* Region C *)
\[Gamma]C1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + C2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], -R0, -(w1 + \[Tau]\[Prime])}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w1 > 0 && w1 + \[Tau]\[Prime] > 0 && w1 + \[Tau]\[Prime] < R0 && \[Tau]0 < R0];
\[Gamma]C1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Gamma]C1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 < R0];
\[Gamma]C = FullSimplify[\[Gamma]C1, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 < R0];
Print[StandardForm["\[Gamma]C:"]];
Print[StandardForm[ToString[\[Gamma]C, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region D *)
\[Gamma]D1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*(ABCD1[\[Tau]\[Prime], \[Theta]\[Prime]] + D2[\[Tau]\[Prime], \[Theta]\[Prime]]), {\[Theta]\[Prime], w1 + \[Tau]\[Prime], R0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w1 > 0 && w1 + \[Tau]\[Prime] > 0 && w1 + \[Tau]\[Prime] < R0 && \[Tau]0 < R0];
\[Gamma]D1 = Integrate[(\[Rho]\[Tau][\[Tau]\[Prime]] & )*\[Gamma]D1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 < R0];
\[Gamma]D = FullSimplify[\[Gamma]D1, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 < R0];
Print[StandardForm["\[Gamma]D:"]];
Print[StandardForm[ToString[\[Gamma]D, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Region E *)
\[Gamma]E1\[Theta][\[Tau]\[Prime]_] = Integrate[\[Rho]\[Theta][\[Theta]\[Prime]]*E1[\[Tau]\[Prime], \[Theta]\[Prime]], {\[Theta]\[Prime], -(w1 + \[Tau]\[Prime]), w1 + \[Tau]\[Prime]}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && w1 > 0 && w1 + \[Tau]\[Prime] > 0 && \[Tau]0 < R0];
\[Gamma]E1 = Integrate[\[Rho]\[Tau][\[Tau]\[Prime]]*\[Gamma]E1\[Theta][\[Tau]\[Prime]], {\[Tau]\[Prime], 0, \[Tau]0}, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 < R0];
\[Gamma]E = FullSimplify[\[Gamma]E1, Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 < R0];
Print[StandardForm["\[Gamma]E:"]];
Print[StandardForm[ToString[\[Gamma]E, CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Average Out-Degree *)

(* Case \[Alpha] : \[Tau]0 > 2 R0 *)
k\[Alpha] = FullSimplify[\[Delta]*(\[Alpha]A + \[Alpha]B + \[Alpha]C + \[Alpha]D + \[Alpha]E), Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > 2*R0 && \[Delta] > 0];
Print[StandardForm["k\[Alpha]:"]];
Print[StandardForm[ToString[k\[Alpha], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case \[Beta] : R0 < \[Tau]0 < 2 R0 *)
k\[Beta] = FullSimplify[\[Delta]*(\[Beta]A + \[Beta]B + \[Beta]C + \[Beta]D + \[Beta]E), Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 > R0 && \[Tau]0 < 2*R0 && \[Delta] > 0];
Print[StandardForm["k\[Beta]:"]];
Print[StandardForm[ToString[k\[Beta], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

(* Case \[Gamma] : \[Tau]0 < R0 *)
k\[Gamma] = FullSimplify[\[Delta]*(\[Gamma]C + \[Gamma]D + \[Gamma]E), Assumptions -> \[Tau]0 > 0 && R0 > 0 && \[Tau]0 < R0 && \[Delta] > 0];
Print[StandardForm["k\[Gamma]:"]];
Print[StandardForm[ToString[k\[Gamma], CharacterEncoding -> "ASCII"]]]; 
Print[StandardForm[""]];

Print[StandardForm["PROGRAM COMPLETED"]];

Exit[];
