package wkde

import (
    "github.com/zpz/matrix.go/dense"
    "github.com/zpz/stats.go"
)

// use_float_slice takes a slice x and required length,
// returns x if it is of correct length,
// returns a newly created slice if x is nil,
// and panic if x is non-nil but has wrong length.
func use_float_slice(x []float64, n int) []float64 {
	if x == nil {
		return make([]float64, n)
	}
	if len(x) != n {
		panic("wrong length")
	}
	return x
}

// use_dense takes a Dense x and required shape,
// returns x if it is of correct shape,
// returns a newly created Dense if x is nil,
// and panic if x is non-nil but has wrong shape.
func use_dense(x *dense.Dense, r, c int) *dense.Dense {
	if x == nil {
		return dense.NewDense(r, c)
	}
	m, n := x.Dims()
	if m != r || n != c {
		panic("wrong shape")
	}
	return x
}


// clone_floats returns a clone of the input slice.
func clone_floats(x []float64) []float64 {
    v := make([]float64, len(x))
    copy(v, x)
    return v
}


// pick_floats returns a subslice with the elements
// at the specified indices.
func pick_floats(x []float64, index []int, y []float64) []float64 {
	y = use_float_slice(y, len(index))
	for i, j := range index {
		y[i] = x[j]
	}
	return y
}

// exclude returns integers between 0 (inclusive) and
// n (exclusive), excluding those in index.
// Does not assume elements in include are sorted
// (otherwise the code can be more efficient).
func exclude(n int, index []int) []int {
	idx := make([]int, n)
	for _, i := range index {
		idx[i] = 1
	}
	k := 0
	for j := 0; j < n; j++ {
		if idx[j] == 0 {
			idx[k] = j
			k++
		}
	}
	return idx[:k]
}


func subcov_xx(
    mat *dense.Dense,
    idx []int,
    out *dense.Dense) *dense.Dense {
    p := len(idx)
    out = use_dense(out, p, p)
    for i, j := range idx {
        pick_floats(mat.RowView(j), idx, out.RowView(i))
    }
    return out
}

func subcov_xy(
    mat *dense.Dense,
    x_idx, y_idx []int,
    out *dense.Dense) *dense.Dense {
    px, py := len(x_idx), len(y_idx)
    out = use_dense(out, px, py)
    for i, j := range x_idx {
        pick_floats(mat.RowView(j), y_idx, out.RowView(i))
    }
    return out
}

// xtAy computes the scalar value that is
//  x^t * A * y
// where 'x' and 'y' are treated as column vectors,
// and '*' is matrix multiplication.
func xtAy(x []float64, A *dense.Dense, y []float64) float64 {
	v := 0.0
	k := len(x)
	for i := 0; i < k; i++ {
		v += stats.FloatDot(A.RowView(i), y) * x[i]
	}
	return v
}
