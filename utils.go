package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
)

type SerializableNeuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type SerializableLayer struct {
	Neurons []SerializableNeuron `json:"neurons"`
}

type SerializableNetwork struct {
	Layers []SerializableLayer `json:"layers"`
}

func (net *Network) SaveWeights(filename string) error {
	sn := SerializableNetwork{}

	for _, layer := range net.Hidden {
		sl := SerializableLayer{}
		for _, neuron := range layer.Neurons {
			sl.Neurons = append(sl.Neurons, SerializableNeuron{
				Weights: neuron.Weights,
				Bias:    neuron.Bias,
			})
		}
		sn.Layers = append(sn.Layers, sl)
	}

	data, err := json.MarshalIndent(sn, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (net *Network) LoadWeights(filename string) error {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		log.Println("Weights file not found, starting fresh.")
		return nil
	}

	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var sn SerializableNetwork
	if err := json.Unmarshal(data, &sn); err != nil {
		return err
	}

	// Copy values back into your network
	for i, sl := range sn.Layers {
		if i >= len(net.Hidden) {
			return fmt.Errorf("mismatch: saved network has more layers (%d) than current network (%d)", len(sn.Layers), len(net.Hidden))
		}
		for j, snNeuron := range sl.Neurons {
			if j >= len(net.Hidden[i].Neurons) {
				return fmt.Errorf("mismatch: layer %d has more neurons in saved file (%d) than in current network (%d)",
					i, len(sl.Neurons), len(net.Hidden[i].Neurons))
			}

			n := net.Hidden[i].Neurons[j]

			if len(n.Weights) != len(snNeuron.Weights) {
				return fmt.Errorf("weight mismatch at layer %d neuron %d: expected %d weights, got %d",
					i, j, len(n.Weights), len(snNeuron.Weights))
			}

			// Safe to assign now
			copy(n.Weights, snNeuron.Weights)
			n.Bias = snNeuron.Bias
		}
	}

	log.Println("Weights loaded successfully.")
	return nil
}

// LoadCSV reads a CSV file and returns numeric features X and target Y.
// It automatically converts categorical (string) features into one-hot encoded vectors.
func LoadCSV(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	if len(records) < 2 {
		return nil, nil, fmt.Errorf("not enough rows in CSV")
	}

	numCols := len(records[0])

	// --- Detect categorical columns ---
	isCategorical := make([]bool, numCols-1) // exclude last column (Y)
	categoryValues := make([]map[string]struct{}, numCols-1)

	for i := 0; i < numCols-1; i++ {
		categoryValues[i] = make(map[string]struct{})
		isCategorical[i] = false
	}

	// Detect categorical by checking if ParseFloat fails
	for _, row := range records[1:] {
		for j := 0; j < numCols-1; j++ {
			_, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				isCategorical[j] = true
				categoryValues[j][row[j]] = struct{}{}
			}
		}
	}

	// --- Create consistent ordering of categories for one-hot encoding ---
	categoryOrder := make([][]string, numCols-1)
	for j := 0; j < numCols-1; j++ {
		if isCategorical[j] {
			for val := range categoryValues[j] {
				categoryOrder[j] = append(categoryOrder[j], val)
			}
			sort.Strings(categoryOrder[j]) // deterministic order
		}
	}

	// --- Build feature matrix X and target vector Y ---
	var X [][]float64
	var Y []float64

	for _, row := range records[1:] {
		var features []float64

		for j := 0; j < numCols-1; j++ {
			if isCategorical[j] {
				oneHot := make([]float64, len(categoryOrder[j]))
				val := row[j]
				for k, cat := range categoryOrder[j] {
					if val == cat {
						oneHot[k] = 1.0
						break
					}
				}
				features = append(features, oneHot...)
			} else {
				val, err := strconv.ParseFloat(row[j], 64)
				if err != nil {
					return nil, nil, fmt.Errorf("invalid numeric value %q in column %d", row[j], j)
				}
				features = append(features, val)
			}
		}

		y, err := strconv.ParseFloat(row[numCols-1], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid target value %q", row[numCols-1])
		}

		X = append(X, features)
		Y = append(Y, y)
	}

	return X, Y, nil
}

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

func Softmax(activations []float64) []float64 {
	expSum := 0.0
	for _, a := range activations {
		expSum += math.Exp(a)
	}
	out := make([]float64, len(activations))
	for i, a := range activations {
		out[i] = math.Exp(a) / expSum
	}
	return out
}
