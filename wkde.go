package wkde

import (
    "math"
    "github.com/zpz/stats.go"
    "github.com/zpz/matrix.go/dense"
)


// Multivariate kernel density estimation based on a weighted sample.
func Wkde(
    // Matrix of sample.
    // Each row is an observation, each col is a variate.
    // The observations are assumed to be in no particular order,
    // and their order is preserved.
    // x should not contain NA, and should all be finite.
    x *dense.Dense,

    // Log relative weights of the observations in x.
    // Do not need to be normalized.
    logwt []float64,

    // Part of the sample points that account for
    // a significant fraction of the total weight
    // is used as component centers in the resultant
    // mixture distribution.
    // These mixture center account for
    // 1 - mix.wt.tol
    // of the total weight (which is 1, if normalized).
    mix_wt_tol float64,
) *Normix {

    nx := x.Rows()

    var idx_mix_ctr []int

    if logwt == nil || mix_wt_tol <= 0.0 {
        idx_mix_ctr = make([]int, nx)
        for i := 0; i < nx; i++ {
            idx_mix_ctr[i] = i
        }
    } else {
        logwt = stats.FloatShift(logwt, -stats.LogSumExp(logwt), nil)

        idx_mix_ctr = lose_weight(logwt, 1 - mix_wt_tol)
            // Identify sample points as mixture component centers.
            // Usually one does not need to use all the sample points
            // as 'center' points if the sample is _weighted_.
            // Instead, it's more efficient to pick a subset
            // of high-weight sample points as 'center' points.
            // This fraction of sample accounts for (1 - mix.wt.tol)
            // of the total weight.
            // This fraction of sample becomes component centers
            // in the resultant mixture distribution.
    }


    // Find local kernels.
    obj_knn := knn(x, logwt, idx_mix_ctr, 2)

    return wkde_optimize(obj_knn)
        // info = z.optim[c('minimum', 'objective')])
}



// Given the skeleton of a kernel density estimator,
// use an optimization procedure to find the optimum bandwidth factor.
// The objective function in the optim algorithm
// is the log density of the testing points, that is,
// the log likelihood of the tuning parameter---the
// bandwidth factor.
// The input object is modified and returned.
func wkde_optimize(obj_knn *Normix) (*Normix) {
    bw_lower := 0.01
    bw_upper := 100.0
        // Starting value and bounds of the bandwidth factor.

    centers := obj_knn.mean
    ndim := centers.Cols()
        // Dimensionality.
    nmix := centers.Rows()
        // Number of mixture components.
    ntest := nmix


    dist_mix_test := dense.NewDense(ntest, nmix)
        // Matrix of (squared) Mahalanobis distance
        // between each mixture component center
        // and each testing point.
        // (The testing points are the same as the mixture component
        // centers.)
        //
        // Matrix of (squared) distances between mixture centers and
        // testing points.
        // Col i is the 'dist kernel' of all testing points
        // to the center of the i'th mixture component, meaning
        //   t(x - y) %*% cov.inv %*% (x - y)
        // Row j is the 'dist kernel' of the jth testing point
        // to the centers of all mixture components.

    cov_logdet_mix := make([]float64, nmix)
        // Log determinant of the cov matrix of each mixture component.

    if obj_knn.kind == FreeCov {
        diff := make([]float64, ndim)
        cov_inv := dense.NewDense(ndim, ndim)

        for imix := 0; imix < nmix; imix++ {
            cov_chol, ok := dense.Chol(obj_knn.cov[imix])
            if !ok {
                panic("cov matrix failed Cholesky")
            }
            cov_logdet_mix[imix] = cov_chol.LogDet()
            cov_chol.Inv(cov_inv)

            for jmix := 0; jmix < nmix; jmix++ {
                v := 0.0
                if jmix != imix {
                    stats.FloatSubtract(centers.RowView(imix),
                        centers.RowView(jmix), diff)
                    v = xtAy(diff, cov_inv, diff)
                }
                dist_mix_test.Set(jmix, imix, v)
            }
        }
    } else {
        diff := make([]float64, ndim)
        cov_inv := dense.NewDense(ndim, ndim)

        cov_chol, ok := dense.Chol(obj_knn.cov[0])
        if !ok {
            panic("cov matrix failed Cholesky")
        }
        fill_float(cov_logdet_mix, cov_chol.LogDet())
        cov_chol.Inv(cov_inv)

        for imix := 0; imix < nmix; imix++ {
            scale := 1.0
            if obj_knn.kind == ScaledCov {
                scale = 1.0 / obj_knn.cov_scale[imix]
            }
            for jmix := 0; jmix < nmix; jmix++ {
                v := 0.0
                if jmix != imix {
                    stats.FloatSubtract(centers.RowView(imix),
                        centers.RowView(jmix), diff)
                    v = xtAy(diff, cov_inv, diff) * scale
                }
                dist_mix_test.Set(jmix, imix, v)
            }
        }
    }


    logwt_mix := obj_knn.logweight
        // Must be normalized.
    logwt_test := logwt_mix

    /*
    # If bandwidth is 'bw' and the weights of the testing points
    # are 'wt.test', then the 'j'th testing point has density
    #
    #  sum(wt.mix * exp(-.5 * (
    #       p.x*log(2*pi) + cov.logdet.mix + p.x*log(bw)
    #       + dist.mix.test[, j]/bw
    #       ))
    #   )
    # =
    #  sum(exp(logwt.mix - .5 * (
    #       p.x*log(2*pi) + cov.logdet.mix + p.x*log(bw)
    #       + dist.mix.test[, j]/bw
    #       ))
    #   )
    #
    # But, we'll skip the 'i'th mixture component if
    # the 'j'th testing point coincides with the center
    # of the 'i'th mixture component. When this happens,
    # the weights
    #   logwt.mix
    # should be replaced by
    #   logwt.mix - log(1 - wt.mix[i])
    #   = logwt.mix - log1p(-wt.mix[i])
    #
    # If the testing point coincides with multiple
    # mixture centers (which is unlikely to happen though),
    # all these mixture components are skipped.
    */


    // Constants to be used in the objective function below.

    logwt_test_normalizer := make([]float64, ntest)
    for i := 0; i < ntest; i++ {
        m := math.Exp(logwt_mix[i])
        // TODO: if m > .50,
        // to much weight is carried by the mixture component center,
        // which is discarded in testing.
        logwt_test_normalizer[i] = math.Log1p(-m)
    }

    a := stats.FloatAddScaled(logwt_mix, cov_logdet_mix, -0.5, nil)
    wt_test := stats.FloatTransform(logwt_test, math.Exp, nil)
    logp := make([]float64, ntest)
    rowspace := make([]float64, nmix)

    // Objective function:
    // negative log density of the test data
    // as a function of the bandwidth factor, 'bw' ('x' in this function).
    f_obj := func(
        x float64, // Bandwidth value to be optimized over. Scalar.
    ) float64 {
        // Using the relation
        //
        //   log(sum(exp(x_i + c))) = c + log(sum(exp(x_i)))
        //
        // log density of the j'th testing point is
        //
        //  log(sum(exp(logwt.mix - logwt.test.normalizer[j]
        //              - .5 * (p.x*log(2*pi)
        //                      + cov.logdet.mix
        //                      + p.x*log(bw)
        //                      + dist.mix.test[, j]/bw
        //                     )
        //             )
        //         )
        //     )
        //  =
        //  log(sum(exp(logwt.mix
        //              - .5 * (cov.logdet.mix + dist.mix.test[, j]/bw)
        //             )
        //         )
        //     ) - .5 * p.x * log(2*pi*bw) - logwt.test.normalizer[j]
        //
        // A number of variables are taken from the parent envir
        // for efficiency.

        // Avoid vectorization in order to save on memory use.
        // The main problem is 'dist_mix_test' can have big dimensions like
        // 8000 x 8000.
        pair := make([]float64, 2)
        for j := 0; j < ntest; j++ {
            dist := dist_mix_test.RowView(j)
            if j > 0 {
                stats.FloatAddScaled(a[:j], dist[:j], -0.5 / x,
                    rowspace[:j])
                pair[0] = stats.LogSumExp(rowspace[:j])
            }
            if j < ntest - 1 {
                stats.FloatAddScaled(a[j+1 :], dist[j+1 :], -0.5 / x,
                    rowspace[j+1 : ])
                    pair[1] = stats.LogSumExp(rowspace[j+1 :])
            }

            if j == 0 {
                logp[j] = pair[1]
            } else if j < ntest - 1 {
                logp[j] = stats.LogSumExp(pair)
            } else {
                logp[j] = pair[0]
            }
        }

        stats.FloatSubtract(logp, logwt_test_normalizer, logp)
        stats.FloatShift(logp, -0.5 * float64(ndim) * math.Log(2.0 * x * math.Pi), logp)
        return - stats.FloatDot(wt_test, logp)
            // Negative log likelihood.
            // The smaller, the better.
    }

    z, converged := stats.Minimize(f_obj, bw_lower, bw_upper, 1.0e-10)

    if (converged) {
        if obj_knn.cov_scale != nil {
            stats.FloatScale(obj_knn.cov_scale, z, obj_knn.cov_scale)
        } else {
            for i := range obj_knn.cov {
                obj_knn.cov[i].Scale(z)
            }
        }
    } else {
        // TODO: log message "optimize failed to converge"
    }

    return obj_knn

    // TODO: return other info regarding the optimization,
    // such as solution, number of function calls, value of objective
    // function at solution, convergence.
}

