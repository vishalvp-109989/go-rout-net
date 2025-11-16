package ml

import (
	"math"
	"math/rand"
	"time"
)

func Transpose(matrix [][]chan float64) [][]chan float64 {
	if len(matrix) == 0 {
		return [][]chan float64{}
	}

	rows := len(matrix)
	cols := len(matrix[0])

	// Create transposed matrix with swapped dimensions
	transposed := make([][]chan float64, cols)
	for i := range transposed {
		transposed[i] = make([]chan float64, rows)
	}

	// Swap rows and columns
	for i := range rows {
		for j := range cols {
			transposed[j][i] = matrix[i][j]
		}
	}

	return transposed
}

func Softmax(activations []float64, T float64) []float64 {
	expSum := 0.0
	for _, a := range activations {
		expSum += math.Exp(a/T)
	}
	out := make([]float64, len(activations))
	for i, a := range activations {
		out[i] = math.Exp(a/T) / expSum
	}
	return out
}

// Convert to One Hot Encoding
func OneHotEncode(x float64, dim int) []float64 {
	targetVector := make([]float64, dim)
	if x >= 0 && x < float64(dim) {
		targetVector[int(x)] = 1.0
	}
	return targetVector
}

func OneHotDecode(v []float64) float64 {
	maxIndex := 0
	maxValue := v[0]
	for i, val := range v {
		if val > maxValue {
			maxValue = val
			maxIndex = i
		}
	}
	return float64(maxIndex)
}

// Shuffle shuffles X and Y using the same permutation.
// Returns the shuffled slices.
func Shuffle(X [][]float64, Y []float64) ([][]float64, []float64) {
	rand.Seed(time.Now().UnixNano())
	n := len(X)

	for i := range X {
		j := rand.Intn(n)

		// swap X
		X[i], X[j] = X[j], X[i]

		// swap Y
		Y[i], Y[j] = Y[j], Y[i]
	}

	return X, Y
}
