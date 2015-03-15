Get["/home/cunningham.wi/local/etc/ToMatlab.m"];

(* Print[$CommandLine];

n = Length[$CommandLine];
Print[n];
k = ToExpression[$CommandLine[[n]]];
Print[k]; *)

kern = ToExpression[$CommandLine[[Length[$CommandLine]]]];
Print@kern;

a = \[Alpha];
b = gr2rom[a];
Print@b;

(* y = gr2rom[Log[\[Alpha]]+Sinh[x]] //ToMatlab *)
y = Log[\[Alpha]] + Sinh[\[Beta]];

Print[StandardForm[ToString[y, CharacterEncoding -> "ASCII"]]];
Print@"Done";

Exit[];
