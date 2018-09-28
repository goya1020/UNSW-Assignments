lambda = 7.2;

syms x real
assume(0<=x & x<=1);
ak1 = (-log(1-x)/lambda);
%ezplot(ak1)


syms y real
assume(0<=x & x<=1);
ak2 = ((1.17-0.75)*y+0.75);
ezplot(ak2)
