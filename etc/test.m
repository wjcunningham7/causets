Get["/home/cunningham.wi/local/etc/ToMatlab.m"];

(* gr2rom[x_] := Module[{s},
	s = ToString[FullForm[x]];
	s = StringReplace[s, RegularExpression["\\\\\[(\\w+)\\]"] -> "$1"];
	ToExpression[s]
	] *)

a = \[Alpha];
b = gr2rom[a];
Print@b;

(* y = gr2rom[Log[\[Alpha]]+Sinh[x]] //ToMatlab *)
y = Log[\[Alpha]] + Sinh[\[Beta]];

Print[StandardForm[ToString[y, CharacterEncoding -> "ASCII"]]];
Print@"Done";

Exit[];
