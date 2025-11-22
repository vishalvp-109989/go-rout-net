package ml

import (
	"math"
	"math/rand"
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

func Softmax(input []float64, temperature float64) []float64 {
	// 1. Find the Maximum value in the input vector
	maxVal := -math.MaxFloat64
	for _, v := range input {
		if v > maxVal {
			maxVal = v
		}
	}

	// 2. Calculate Exponentials subtracting maxVal (Stable Step)
	// e^(x - max) is mathematically equivalent for Softmax but prevents overflow
	output := make([]float64, len(input))
	sum := 0.0

	for i, v := range input {
		// v - maxVal will always be <= 0, so Exp will result in 0.0 to 1.0
		// It will NEVER be Infinity.
		e := math.Exp((v - maxVal) / temperature)
		output[i] = e
		sum += e
	}

	// 3. Normalize
	for i := range output {
		output[i] = output[i] / sum
	}

	return output
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

// ShuffleGeneric shuffles two slices of any type (T and U) in sync.
// It returns an error if slice lengths do not match.
func Shuffle[T any, U any](x []T, y []U) error {
	if len(x) != len(y) {
		panic("slices must be of the same length")
	}

	// Use the standard library's shuffle helper
	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})

	return nil
}
