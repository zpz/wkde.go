package wkde

import (
	"github.com/zpz/stats.go"
	"math"
)

/*
   Suppose there is a reference/target density f,
   and a sample X has been drawn from an approximating
   density g, then
     weight = f(x) / g(x)
   ie, the importance weight of sampling f via g.
   If 'log' is TRUE, then
     weight = log(f(x)) - log(g(x))

   In practice it is often the case that one or both of
   f and g is un-normalized. Let the un-normalized
   densities be denoted by f' and g'.

   Let f/g = a * f'/g'
   Because
     1 = int f dx = a * int (f'/g') g dx
   We have
     a = 1 / (int (f'/g') g dx)
   Given a random sample from g, and let
     w_i = f'_i / g'_i
   then an estimate of a is
     n / sum(w_i)
   hence
     f/g = a * f'/g' = n * w_i / sum(w_i)

   Let v_i = w_i / sum(w_i)


   --- 1 ---
   Entropy of sample weights relative to uniformity.

     -(1 / log(n)) * sum(w_i * log(w_i))

   This metric approaches 1 as the weights approach uniform.
   This entropy is equivalent to an estimate (for it is based on
   a random sample, X) of the Kullback-Leibler divergence
   between the two distributions with 'f' being the reference one.

   If all weights are equal to (1/n), this entropy is 1.
   (This is the extreme case of balanced weights.)
   If one weight is 1 and all others are 0, this entropy is 0.
   (This is the extreme case of unbalanced weights.)

   Reference:

   West, 1993,
   "Approximating posterior distributions by mixture",
   J. R. Stat. Soc. B, 55(2).

   Kullback and Leibler, 1951,
   Ann. Math. Statist. 22: 79--86.

   Handcock and Morris, 1998,
   "Relative distribution methods",
   Sociological Methodology, 28(1): 53--97.
*/

func logweight_entropy(logwt []float64) float64 {
	n := len(logwt)
	z := stats.LogSumExp(logwt)
	// TODO: check if logwt is already normalized in all its use cases.
	// If yes, then this can be skipped.

	res := 0.0
	for _, v := range logwt {
		v -= z
		res += v * math.Exp(v)
	}

	return -res / math.Log(float64(n))
}
