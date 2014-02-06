package stats

import (
	"github.com/pmylund/sortutil"
	"github.com/zpz/matrix.go/dense"
	"github.com/zpz/stats.go"
	"log"
	"math"
	"sort"
)

type NormixKind int

const (
	// All mixture components have the same cov matrix.
	SharedCov NormixKind = iota

	// The cov matrices of all mixture components
	// are scaled versions of the same cov matrix.
	// Each mixture component has its own scaling factor.
	ScaledCov

	// Each mixture component has a distinct cov matrix.
	FreeCov
)


// Normix defines a normal mixture distribution.
type Normix struct {
	kind NormixKind

	// Log weight of each mixture component.
	// Length is the number of mixture components.
	logweight []float64

	// Each row is the mean of one mixture component.
	// Each col is one dimension.
	mean *dense.Dense

    // Cov matrix for each mixture component.
	cov []*dense.Dense

	// If kind is SharedCov, cov has length 1, cov_scale is nil.
	// If kind is ScaledCov, cov has length 1, cov_scale contains
	// scaling factor of the cov matrix for each mixture component.
	// If kind is FreeCov, cov has as many elements as there are mixture
	// components, and cov_scale is nil.
	cov_scale []float64
}

// NewNormix creates a Normix object and returns a pointer to it.
// It allocates memory for all fields,
// which will be populated later.
func NewNormix(n_dim, n_mix int, kind NormixKind) *Normix {
	if n_dim < 1 || n_mix < 1 {
		panic("NewNormix: positive arguments expected for n_dim and n_mix")
	}

	var mix Normix

	mix.kind = kind
	mix.logweight = make([]float64, n_mix)
	mix.mean = dense.NewDense(n_mix, n_dim)
	mix.cov_scale = nil

	switch kind {
	case SharedCov:
		mix.cov = make([]*dense.Dense, 1)
        mix.cov[0] = dense.NewDense(n_dim, n_dim)
	case ScaledCov:
		// Usually this is not used when n_dim is 1,
		// but it is allowed.
		mix.cov = make([]*dense.Dense, 1)
        mix.cov[0] = dense.NewDense(n_dim, n_dim)
		mix.cov_scale = make([]float64, n_mix)
	case FreeCov:
		mix.cov = make([]*dense.Dense, n_mix)
        for i := range mix.cov {
            mix.cov[i] = dense.NewDense(n_dim, n_dim)
        }
	default:
		panic("NewNormix: unrecognized value for kind")
	}

	return &mix
}


// dim returns the number of dimensions of the normal mixture distribution.
func (mix *Normix) Dim() int {
	return mix.mean.Cols()
}

// size returns the number of mixture components in the normal mixture
// distribution.
func (mix *Normix) Size() int {
	return len(mix.logweight)
}




// Pooled mean and variance of a random vector represented by a
// mixture---
//   Let there be 'n' mixtures for a k-vector
//   with weights a_i and component means m_i
//   and component cov matrices v_i,
//   then the overall mean and cov are
//     m = a_1 m_1 + ... + a_n m_n
//     v = a_1(v_i + m_i m_i^T) + ... + a_n(v_n + m_n m_n^T) - m m^T
//
// This function does not assume normality of the mixtures.


// Density computes the pdf of each row of x in the Normix mix.
func (mix *Normix) Density(x *dense.Dense, out []float64) []float64 {
	ndim := mix.Dim()
	assert(x.Cols() == ndim, "Wrong shape for input x")

	nx := x.Rows()

	zz := mix.density_stats(x, nil)

	nmix := mix.Size()

	if nmix == 1 {
		out = zz.GetData(out)
	} else {
		for imix := 0; imix < nmix; imix++ {
			stats.FloatShift(zz.RowView(imix), mix.logweight[imix], zz.RowView(imix))
		}
		lw := make([]float64, nmix)
		for ix := 0; ix < nx; ix++ {
			out[ix] = stats.LogSumExp(zz.GetCol(ix, lw))
		}
	}

	return stats.FloatTransform(out, math.Exp, out)
}

// Random generates n random samples from the normal mixture
// distribution and returns the sample in a slice,
// one case after another.
func (mix *Normix) Random(n int, out *dense.Dense) *dense.Dense {
	ndim := mix.dim()
	out = use_dense(out, n, ndim)

	mixidx := stats.LogweightedSample(mix.logweight, n, nil)
	sort.Ints(mixidx)

	//cov_mat := dense.NewDense(ndim, ndim)

	switch mix.Kind() {
	case SharedCov:
		mvn := NewMVN(make([]float64, ndim), mix.cov[0])

		mvn.Random(n, &RNG{}, out)

		for i, idx := range mixidx {
			z := out.RowView(i)
			stats.FloatAdd(z, mix.mean.RowView(idx), z)
		}

	case ScaledCov:
		mvn := NewMVN(make([]float64, ndim), mix.cov[0])

		mvn.Random(n, &RNG{}, out)

		for i, idx := range mixidx {
			z := out.RowView(i)
			stats.FloatScale(z, math.Sqrt(mix.cov_scale[idx]), z)
			stats.FloatAdd(z, mix.mean.RowView(idx), z)
		}

	case FreeCov:
		for iz := 0; iz < n; {
			imix := mixidx[iz]
			nmix := 1
			iz++
			for ; iz < n && mixidx[iz] == imix; iz++ {
				nmix++
			}
			// When the weights of the mixture components are
			// highly uneven, it's common that a small number
			// of the high-weight components are repeatedly
			// used in generating samples.
			// We find the number of samples each mixture
			// component generates, so that these samples
			// are generated at once.

			mvn := NewMVN(mix.mean.RowView(imix), mix.cov[imix])
			z := out.SubmatrixView(iz-nmix, 0, nmix, ndim)
			mvn.Random(nmix, &RNG{}, z)
		}
	}

	return out
}

// Marginal returns the marginal distribution, as a Normix,
// of the dimensions specified by dims.
func (mix *Normix) Marginal(dims []int) *Normix {
	ndim := len(dims)
	nmix := mix.Size()
	kind := mix.Kind()

	assert(ndim > mix.Dim(), "too many elements in dims")

	// TODO: check that dims does not contain duplicate elements.

	out := NewNormix(ndim, nmix, kind)

	// Get out.logweight.
	copy(out.logweight, mix.logweight)

	// Get out.mean.
	for imix := 0; imix < nmix; imix++ {
		pick_floats(mix.mean.RowView(imix), dims, out.mean.RowView(imix))
	}

	// Get out.cov.
	if kind == FreeCov {
		for imix := 0; imix < nmix; imix++ {
            subcov_xx(mix.cov[imix], dims, out.cov[imix])
		}
	} else {
        subcov_xx(mix.cov[0], dims, out.cov[0])
	}

	// Get out.cov_scale.
	if kind == ScaledCov {
		copy(out.cov_scale, mix.cov_scale)
	}

	return out
}

// Conditional derives conditional density of some dimensions in a
// normal mixture,
// given observed values for the other dimensions.
func (mix *Normix) Conditional(
	data []float64,
	// Data vector.
	dims []int,
	// Values in data correspond to these dimensions in mix.
	wt_tol float64,
	// In the resultant mixture density,
	// components with highest weights that collectively
	// account for (1 - wt_tol) of the total weight are kept.
	// Use 0 if you don't know better.
) *Normix {

	assert(len(data) == len(dims), "dimensionality mismatch")

	//==================================================
	// Calculate likelihoods of the data in its marginal
	// distribution.

	marginal_pdf := mix.Marginal(dims)
	loglikely := marginal_pdf.density_stats(
		dense.DenseView(data, 1, len(dims)), nil)

	//========================================
	// Update weight of each mixture component
	// to take into account the likelihoods.

	logwt := stats.FloatAdd(loglikely.DataView(), mix.logweight, nil)
	logintlikely := stats.LogSumExp(logwt)
	// Log integrated likelihood.
	stats.FloatShift(logwt, -logintlikely, logwt)
	// Normalized; now sum(exp(logwt)) = 1.

	// Screen the mixture components and discard those
	// with negligible weights.

	var idx_keep []int

	if len(logwt) > 1 && wt_tol > 0 {
		idx_keep := lose_weight(logwt, 1-wt_tol)

		if len(idx_keep) < len(logwt) {
			logwt = pick_floats(logwt, idx_keep, nil)
            lse := stats.LogSumExp(logwt)
			total_wt := math.Exp(lse)

			log.Println("keeping",
				len(idx_keep), "of", mix.Size(),
				"components for a total weight of",
				total_wt)

			stats.FloatShift(logwt, -lse, logwt)
			// Normalize so that weights sum to 1.
		}
	} else {
		idx_keep = make([]int, len(logwt))
		for i, _ := range idx_keep {
			idx_keep[i] = i
		}
	}

	//====================================
	// Compute conditional mean and cov.

	n_y := len(dims)
	dims_y := dims
	// 'y' indicates the conditioning dimensions and data.

	n_x := mix.Dim() - n_y
	dims_x := exclude(mix.Dim(), dims_y)
	// 'x' indicates the conditioned, i.e. target,
	// dimensions.

	mix_x := NewNormix(n_x, len(logwt), mix.kind)
	// Conditional distribution.
	copy(mix_x.logweight, logwt)

	sigma_y := dense.NewDense(n_y, n_y)
	// Cov matrix between dimensions dims_y.

	sigma_xy := dense.NewDense(n_x, n_y)
	// Cov matrix between dims_x and dims_y.

	if mix.kind == FreeCov {
		mu_y := make([]float64, n_y)

		for idx_new, idx_old := range idx_keep {
			mu := mix.mean.RowView(idx_old)
			sigma := mix.cov[idx_old]
			mu_x := mix_x.mean.RowView(idx_new)
			sigma_x := mix_x.cov[idx_new]

			pick_floats(mu, dims_y, mu_y)

            subcov_xx(sigma, dims_y, sigma_y)
            subcov_xy(sigma, dims_x, dims_y, sigma_xy)

			conditional_normal(
				data, mu_y,
				sigma_y, sigma_xy, nil,
				mu_x, sigma_x)

			for i, idx := range dims_x {
				mu_x[i] += mu[idx]
			}

            for i, ii := range dims_x {
                for j, jj := range dims_x {
                    v := sigma_x.Get(i, j)
                    sigma_x.Set(i, j, v + sigma.Get(ii, jj))
                }
            }
		}

	} else {
		if mix_x.kind == ScaledCov {
			pick_floats(mix.cov_scale, idx_keep, mix_x.cov_scale)
		}

		sigma := mix.cov[0]
		sigma_x := mix_x.cov[0]

        subcov_xx(sigma, dims_y, sigma_y)
        subcov_xy(sigma, dims_x, dims_y, sigma_xy)

		mu_x_delta := make([]float64, n_x)

		conditional_normal(
			data, make([]float64, n_y),
			sigma_y, sigma_xy, nil,
			mu_x_delta, sigma_x)
		// sigma_xy is changed in this function,
		// and the new value is useful below.
		A := sigma_xy

        for i, ii := range dims_x {
            for j, jj := range dims_x {
                v := sigma_x.Get(i, j)
                sigma_x.Set(i, j, v + sigma.Get(ii, jj))
            }
        }

		mu_y := make([]float64, n_y)

		for idx_new, idx_old := range idx_keep {
			mu := mix.mean.RowView(idx_old)
			mu_x := mix_x.mean.RowView(idx_new)

			pick_floats(mu, dims_y, mu_y)
			dense.Mult(dense.DenseView(mu_y, 1, n_y), A,
				dense.DenseView(mu_x, 1, n_x))

			stats.FloatSubtract(mu_x_delta, mu_x, mu_x)

			for i, idx := range dims_x {
				mu_x[i] += mu[idx]
			}
		}
	}

	return mix_x
}

// density_stats computes the log-density of the data in each mixture component.
// Input x contains one 'observation' or 'case' per row.
// Upon return, row i of out contains log-densities of every row of x
// in the i-th mixture component of mix.
func (mix *Normix) density_stats(x *dense.Dense, out *dense.Dense) *dense.Dense {
	ndim, nmix := mix.Dim(), mix.Size()
	nx := x.Rows()

	assert(x.Cols() == ndim, "input x has wrong shape")
	out = use_dense(out, nmix, nx)

	if ndim == 1 {
		var xx []float64
		if x.Contiguous() {
			xx = x.DataView()
		} else {
			xx = x.GetData(nil)
		}
		switch mix.kind {
		case SharedCov:
			sd := math.Sqrt(mix.cov[0].Get(0, 0))
			for imix := 0; imix < nmix; imix++ {
				NewNormal(mix.mean.Get(imix, 0), sd).
					Density(xx, out.RowView(imix))
			}
		case ScaledCov:
			v := mix.cov[0].Get(0, 0)
			for imix := 0; imix < nmix; imix++ {
				NewNormal(mix.mean.Get(imix, 0),
					math.Sqrt(v*mix.cov_scale[imix])).
					Density(xx, out.RowView(imix))
			}
		case FreeCov:
			for imix := 0; imix < nmix; imix++ {
				NewNormal(mix.mean.Get(imix, 0),
					math.Sqrt(mix.cov[imix].Get(0, 0))).
					Density(xx, out.RowView(imix))
			}
		}
		return out
	}


	if mix.kind == FreeCov {
		for imix := 0; imix < nmix; imix++ {
			mvn := NewMVN(mix.mean.RowView(imix), mix.cov[imix])
			mvn.Density(x, out.RowView(imix))
		}
	} else {
		// Create a mvn with zero mean.
		mvn := NewMVN(make([]float64, ndim), mix.cov[0])

		// Subtract mean from all data so that their densities
		// are computed using the zero-mean distribution above.
		xx := dense.NewDense(nx*nmix, ndim)
		for imix := 0; imix < nmix; imix++ {
			xxview := xx.SubmatrixView(imix*nx, 0, nx, ndim)
			dense.Copy(xxview, x)
			for row := 0; row < nx; row++ {
				stats.FloatSubtract(xxview.RowView(row),
                    mix.mean.RowView(imix), xxview.RowView(row))
			}
		}

		if mix.kind == SharedCov {
			if out.Contiguous() {
				mvn.Density(xx, out.DataView())
			} else {
				out.SetData(mvn.Density(xx, nil))
			}
		} else { // ScaledCov
            cov_mat := mix.cov[0]
            chol, ok := dense.Chol(cov_mat)
            if !ok {
                panic("Cholesdky failed on cov matrix")
            }
            cov_inv := chol.Inv(nil)
            cov_det := chol.Det()

            // Squared Mahalanobis distance.
            nxx := xx.Rows()
            dist := make([]float64, nxx)
            for i := 0; i < nxx; i++ {
                dist[i] = xtAy(xx.RowView(i), cov_inv, xx.RowView(i))
            }

			for imix := 0; imix < nmix; imix++ {
				res := out.RowView(imix)
				cov_scale := mix.cov_scale[imix]
				coef := 1.0 / math.Sqrt(
					math.Pow(2*math.Pi*cov_scale, float64(ndim))*
						cov_det)
				stats.FloatScale(dist[imix*nx:(imix+1)*nx], -0.5/cov_scale, res)
				stats.FloatTransform(res, math.Exp, res)
				stats.FloatScale(res, coef, res)
			}
		}
	}

	return out
}

// lose_weight takes a slice of log weights
// and returns the indices of the high weights such that
// their total reaches a specified fraction of the total of all weights.
// The intended use case is to discard a large number of low-weight
// elements in a highly skewed weighted sample.
// The returned indices are in order of decreasing weights except
// when keep_total == 1, in which case the return is simply
// the full list of indices.
func lose_weight(
	logweight []float64,
	// Log weights; does not need to be normalized.
	args ...float64,
	// Up to two optional arguments;
	// the first would be keep_total and the second would be keep_tol.
	// keep_total: high weights that account for at least this fraction
	// of the total weights will be kept.
	// keep_tol: keep all weights that are larger than this fraction of
	// the max weight in the input.
	// As a result, if all weights are equal, then the entire sample is
	// retained.
	// Example values for keep_total and keep_tol may be 0.99 and 0.1,
	// respectively.
) []int {

	keep_total := 0.99
	keep_tol := 0.1
	if len(args) > 0 {
		keep_total = args[0]
		if len(args) > 1 {
			keep_tol = args[1]
		}
	}

	if keep_total >= 0.999999 {
		// Simply return all.
		z := make([]int, len(logweight))
		for i := range z {
			z[i] = i
		}
		return z
	}

	// Make a copy; don't modify the input.
	logwt := append([]float64{}, logweight...)

	// Normalize, such that sum(exp(logwt)) == 1.
	FloatShift(logwt, -LogSumExp(logwt), logwt)

	// Get indices listing the weights in descending order.
	idx_ordered := FloatOrder(logwt, nil)
	sortutil.Reverse(idx_ordered)

	n := len(logwt)

	// Get number of entries to keep according to keep_total.
	k := 1
	sum := 0.0
	for _, idx := range idx_ordered {
		sum += math.Exp(logwt[idx])
		if sum < keep_total {
			k++
		} else {
			break
		}
	}
	if k > n {
		k = n
	}

	// Get additional number of entries to keep according to keep_tol.
	if k < n {
		bar := math.Log(keep_tol) + logwt[idx_ordered[0]]
		for k < n {
			if logwt[idx_ordered[k]] > bar {
				k++
			} else {
				break
			}
		}
	}

	if k > n/2 {
		return idx_ordered[:k]
	}

	idx := make([]int, k)
	copy(idx, idx_ordered[:k])
	return idx
}


func conditional_normal(
	y []float64, // data
	mu_y []float64, // mean
	sigma_yy, sigma_xy *dense.Dense, // cov_yy, cov_xy
	A *dense.Dense, // workspace; if nil, sigma_xy will be overwritten
	mu_x_delta, sigma_xx_delta []float64,
	// Conditional mean of x is original mean of x
	// plus mu_x_delta;
	// conditional cov of x (the lower-tri elements)
	// is original cov of x plus sigma_xx_delta.
) {
	// mu_{x|y} = mu_x + S_{xy} Inv(S_{yy}) (y - mu_y)
	// S_{x|y} = S_{xx} - S_{xy} Inv(S_{yy}) S_{yx}

	dimx := sigma_xy.Rows()
	dimy := len(mu_y)
	assert(
		sigma_yy.Rows() == dimy &&
			sigma_yy.Cols() == dimy &&
			sigma_xy.Cols() == dimy &&
			len(y) == dimy &&
			len(mu_x_delta) == dimx &&
			len(sigma_xx_delta) == (dimx*dimx+dimx)/2,
		"dimensions mismatch")

	if A == nil {
		A = sigma_xy
	} else {
		assert(A.Rows() == dimx && A.Cols() == dimy,
			"A have wrong shape")
		dense.Copy(A, sigma_xy)
	}

	if chol, ok := dense.Chol(sigma_yy); ok {
		A = chol.SolveR(A)
		// Now A is sigma_xy * Inv(sigma_yy)
	} else {
		panic("Cholesky failed on cov matrix")
	}

	dy := FloatSubtract(y, mu_y, nil)
	dense.Mult(A, dense.DenseView(dy, dimy, 1),
		dense.DenseView(mu_x_delta, dimx, 1))

	k := 0
	for col := 0; col < dimx; col++ {
		for row := col; row < dimx; row++ {
			// Need the element (row, col) of A %*% sigma_yx,
			// which is the dot product of the row-th row of A
			// and the col-th row of sigma_xy.
			sigma_xx_delta[k] = -FloatDot(A.RowView(row),
				sigma_xy.RowView(col))
			k++
		}
	}
}
