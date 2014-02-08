package wkde

import (
	"github.com/zpz/matrix.go/dense"
	"github.com/zpz/stats.go"
	"math"
)

func rowheads(x *dense.Dense) [][]float64 {
	n := x.Rows()
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = x.RowView(i)
	}
	return out
}

// Kth nearest neighbor according to Mahanalobis distance.
func knn(
	// Each row of x is a 'case' or 'observation';
	// each col is a 'variate' or 'dimension'.
	x *dense.Dense,

	// Log relative weights of the observations in x.
	// Must be all finite values.
	// Do not need to be normalized.
	// If nil, taken to mean equal weights.
	logwt []float64,

	// Indices of 'x' that serve as centers.
	// If nil, taken to mean all observations,
	// that is, 0 : x.Rows().
	// These indices should not have duplicates;
	// their values all fall in [0, x.Rows());
	// these are not checked.
	kernel_idx []int,

	// In a localized kernel density estimator,
	// the cov matrix placed centered at each sample point
	// is calculated based on a fraction of the entire sample.
	// The local fractions considered are
	//   1 / 2^(0 : localization_depth),
	// in that order.
	// Must be >= 0.
	// A good default value is 2.
	localization_depth int,
) *Normix {

	// Function that takes the log weights (in logwt)
	// and outputs a flattened set of weights,
	// logwt_cov, to be used in
	// computing weighted empirical cov matrices.
	// For example, one may flatten the original weights by taking
	// an exponential:
	//
	//   weight ^ alpha
	//
	// where alpha should be small
	// if the weights are very disparate;
	// and close to 1 if the weights are near uniform
	// (i.e. all equal).
	//
	// The following version takes the entropy of the
	// original weights as alpha.
	//
	// Another measure I explored, but abandoned, is
	//   z <- sin(z^1 * pi - .5*pi) / 2 + .5
	// 'z' is on [0,1]
	// 'z^alpha * pi - .5 * pi' is on [-pi/2, pi/2]
	// The resultant 'z' is on [0,1]
	// Tune 'alpha' to adjust the shape of the transform.
	f_logwt_cov := func(x []float64) []float64 {
		// x is log weight
		out := make([]float64, len(x))
		stats.FloatScale(x, logweight_entropy(x), out)
		stats.FloatShift(out, -stats.LogSumExp(out), out)
		// Must be normalized like this.
		return out
	}

	// Dimensionality.
	px := x.Cols()

	// Sample size.
	nx := x.Rows()

	// Transpose of x, convenient for some computations.
	// Each row is one dimension; each col is one observation.
	xt := dense.T(x, nil)

	// One slice for each dimension.
	dim_slices := rowheads(xt)

	// Global cov matrix, and its inverse,
	// between all observations, weighted if logwt is present.
	var global_cov, global_cov_inv *dense.Dense

	if logwt == nil {
		global_cov = stats.FloatCov(dim_slices, nil)
	} else {
		// Normalize, that is, make sum to 1.
		// Do not change the input; make a copy.
		logwt = stats.FloatShift(logwt, -stats.LogSumExp(logwt), nil)

		// Weights used specifically for computing cov matrix.
		// Dampened to avoid singularity or overly heavy influence of
		// a small number of observations.
		logwt_cov := f_logwt_cov(logwt)

		// In-place transformation.
		wt_cov := stats.FloatTransform(logwt_cov, math.Exp, logwt_cov)

		// Weighted cov matrix.
		global_cov = stats.FloatWeightedCov(dim_slices, wt_cov, nil)
	}

	if kernel_idx == nil || len(kernel_idx) < 1 {
		// Use all observations as mixture centers.
		kernel_idx = make([]int, nx)
		for i := range kernel_idx {
			kernel_idx[i] = i
		}
	}

	nmix := len(kernel_idx)

	global_cov_chol, ok := dense.Chol(global_cov)
	if !ok {
		panic("empirical cov matrix is not invertible")
	}
	global_cov_inv = global_cov_chol.Inv(nil)

	var obj *Normix
	if localization_depth < 1 {
		obj = NewNormix(px, nmix, SharedCov)
	} else {
		obj = NewNormix(px, nmix, FreeCov)
	}

	if logwt == nil {
		// Create equal weights, normalized.
		fill_float(obj.logweight, -math.Log(float64(nmix)))
	} else {
		pick_floats(logwt, kernel_idx, obj.logweight)
		if len(kernel_idx) < nx {
			// Normalize weights.
			stats.FloatShift(obj.logweight, -stats.LogSumExp(obj.logweight),
				obj.logweight)
		}
	}

	for i, idx := range kernel_idx {
		// Copy mixture centers into Normix obj.
		copy(obj.mean.RowView(i), x.RowView(idx))
	}

	if localization_depth < 1 {
		dense.Copy(obj.cov[0], global_cov)
		return obj
	}

	// 'flat' cov matrix, and its inverse, b/t all observations,
	// ignoring logwt.
	var global_flatcov, global_flatcov_inv *dense.Dense

	if logwt == nil {
		global_flatcov = global_cov
		global_flatcov_inv = global_cov_inv
	} else {
		// Cov matrix between observations, not weighted.
		global_flatcov = stats.FloatCov(dim_slices, nil)
		chol, ok := dense.Chol(global_flatcov)
		if !ok {
			panic("cov matrix of x is not positive definite")
		}
		global_flatcov_inv = chol.Inv(nil)
	}

	neighbors := make([]int, nx)
	dist := make([]float64, nx)
	diff := make([]float64, px)
	cov_local := dense.NewDense(px, px)

	for i_kernel, i_obs := range kernel_idx {
		center := x.RowView(i_obs)

		// At the beginning, consider all observations as neighbors.
		neighbors = neighbors[:nx]
		for i := 0; i < nx; i++ {
			neighbors[i] = i
		}

		// Inverse cov matrix between all neighbors.
		flatcov_inv := global_flatcov_inv

		for i_depth := 1; i_depth <= localization_depth; i_depth++ {

			// Squared Mahalanobis distance from
			// center to each neighbor row in x.
			// Use the un-weighted cov matrix to calculate
			// Mahalanobis distance.
			dist = dist[:len(neighbors)]
			for i, j := range neighbors {
				stats.FloatSubtract(x.RowView(j), center, diff)
				dist[i] = xtAy(diff, flatcov_inv, diff)
			}

			dd := stats.FloatMedian(dist)

			// Identify neighbors within original neighbors.
			// Because we filter by the median distance,
			// the set of neighbors is halved.
			n_neighbors := 0
			for i, d := range dist {
				if d < dd {
					neighbors[n_neighbors] = neighbors[i]
					n_neighbors++
				}
			}
			neighbors = neighbors[:n_neighbors]

			// Reuse 'xt' to store the neighbor observations.
			for i, idx := range neighbors {
				xt.SetCol(i, x.RowView(idx))
			}
			dim_slices := rowheads(xt.SubmatrixView(0, 0, px, n_neighbors))

			// Weighted cov matrix between the newly identified
			// (reduced set of) neighbors.
			// TODO: if this is not the last depth,
			// this computation of cov_local could be useless, hence wasteful.
			if logwt == nil {
				cov_local = stats.FloatCov(dim_slices, cov_local)
			} else {
				logwt_local := f_logwt_cov(pick_floats(logwt, neighbors, nil))
				// TODO: wasteful in memory here.

				// In-place transform.
				wt_local := stats.FloatTransform(logwt_local, math.Exp, logwt_local)

				cov_local = stats.FloatWeightedCov(dim_slices, wt_local, cov_local)
			}

			// FIXME
			// Is there a simple way to check positive-definiteness
			// without attempting Cholesky?

			chol, ok := dense.Chol(cov_local)
			if ok {
				dense.Copy(obj.cov[i_kernel], cov_local)

				if i_depth < localization_depth {
					// Compute inverse unweighted cov matrix
					// to be used in next iteration for computing
					// Mahalanobis distances.
					// Re-use cov_local.
					if logwt == nil {
						flatcov_inv = chol.Inv(cov_local)
					} else {
						cov_local = stats.FloatCov(dim_slices, cov_local)
						if chol, ok := dense.Chol(cov_local); ok {
							flatcov_inv = chol.Inv(cov_local)
						} else {
							break
							// FIXME: is it possible that weighted
							// cov_local has passed Cholesky whereas
							// this unweighted one fails?
						}
					}
				}
			} else {
				if i_depth == 1 {
					dense.Copy(obj.cov[i_kernel], global_cov)
				}
				// Otherwise, the cov_local saved in last depth
				// iteration is kept.
				break
			}
		}
	}

	return obj
}
