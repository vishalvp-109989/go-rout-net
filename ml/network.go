package ml

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"
)

// Loss function constants (enum)
const (
	MSE                       = iota // Mean Squared Error (Regression)
	CATEGORICAL_CROSS_ENTROPY        // Multi-Class Classification
	BINARY_CROSS_ENTROPY             // Binary Classification
)

type Network struct {
	InputLayer  *Layer
	Hidden      []*Layer
	OutputLayer *Layer
}

type TrainingConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	LossFunction int
	KClasses     int
	VerboseEvery int
}

type SerializableNeuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type SerializableLayer struct {
	// For Dense layers
	Neurons []SerializableNeuron `json:"neurons,omitempty"`
	// For Embedding layers (2D matrix of weights)
	Embeddings [][]float64 `json:"embeddings,omitempty"`
}

type SerializableNetwork struct {
	Layers []SerializableLayer `json:"layers"`
}

// NewNetwork builds: InputLayer -> Layer(defs[1]) -> Layer(defs[2]) -> ... -> Layer(defs[len-1]) -> OutputLayer
func NewNetwork(defs ...LayerDef) *Network {
	if len(defs) < 2 {
		panic("Provide at least an input spec (Embedding/Dense) and one Dense layer.")
	}

	var iLayer *Layer
	var prevErrs, prevIns [][]chan float64
	var hidden []*Layer
	loopStart := 0

	// 1. Handle the first layer (which defines the InputLayer structure)
	def0 := defs[0]

	if def0.IsEmbedding {
		// If Embedding: InputLayer must be 1 row x InputLen columns (e.g., 1x5)
		// InputLayer will send the 5 word indices (x1..x5)
		iLayer = NewInputLayer(1, def0.InputNeurons)

		// Create the Embedding layer
		l := NewEmbedLayer(iLayer.ErrsFromNext, iLayer.InsToNext, def0, defs[1])
		hidden = append(hidden, l)

		// Advance for the next layer (Dense)
		prevErrs = l.ErrsFromNext
		prevIns = l.InsToNext
		loopStart = 1 // Start loop from defs[1]

	} else if !def0.HasInputSpec {
		// Original Dense check
		panic("First Dense must be called with two args: Dense(m, inputDim)")

	} else {
		// Original Dense Input Setup: m=def0.Neurons, n=def0.InputNeurons
		iLayer = NewInputLayer(def0.Neurons, def0.InputNeurons)
		prevErrs = iLayer.ErrsFromNext
		prevIns = iLayer.InsToNext
		loopStart = 0 // Start loop from defs[0]
	}

	// 2. Create subsequent Dense layers
	// Start from the next definition (defs[loopStart]) up to the second-to-last layer.
	for i := loopStart; i < len(defs)-1; i++ {
		l := NewLayer(prevErrs, prevIns, defs[i+1])
		hidden = append(hidden, l)

		// advance the prevs to this layer's outputs for the next iteration
		prevErrs = l.ErrsFromNext
		prevIns = l.InsToNext
	}

	// 3. Last layer (using the last definition in the slice)
	lastDef := defs[len(defs)-1]

	// Assuming a simplified output layer structure for demonstration
	// In a real implementation, the last layer would use NewLayer with a proper def.
	finalDef := LayerDef{
		Neurons:     1,
		Activation:  lastDef.Activation,
		Gradient:    lastDef.Gradient,
		Initializer: lastDef.Initializer,
	}
	l := NewLayer(prevErrs, prevIns, finalDef)
	hidden = append(hidden, l)

	return &Network{
		InputLayer:  iLayer,
		Hidden:      hidden,
		OutputLayer: hidden[len(hidden)-1],
	}
}

func (nw *Network) SaveWeights(filename string) error {
	sn := SerializableNetwork{}

	for _, layer := range nw.Hidden {
		sl := SerializableLayer{}

		if layer.EmbeddingNeuron != nil {
			// Case 1: Embedding Layer
			sl.Embeddings = layer.EmbeddingNeuron.Embeddings
		} else {
			// Case 2: Dense Layer (Iterate over neurons)
			for _, neuron := range layer.Neurons {
				sl.Neurons = append(sl.Neurons, SerializableNeuron{
					Weights: neuron.Weights,
					Bias:    neuron.Bias,
				})
			}
		}
		sn.Layers = append(sn.Layers, sl)
	}

	data, err := json.MarshalIndent(sn, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (nw *Network) LoadWeights(filename string) error {
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

	if len(sn.Layers) != len(nw.Hidden) {
		return fmt.Errorf("mismatch: saved network has %d layers, current network has %d", len(sn.Layers), len(nw.Hidden))
	}

	// Copy values back into your network
	for i, sl := range sn.Layers {
		currentLayer := nw.Hidden[i]

		if currentLayer.EmbeddingNeuron != nil {
			// Case 1: Embedding Layer
			if len(sl.Embeddings) == 0 {
				return fmt.Errorf("mismatch: layer %d is an Embedding layer but no 'embeddings' found in saved file", i)
			}
			if len(sl.Embeddings) != len(currentLayer.EmbeddingNeuron.Embeddings) ||
				(len(sl.Embeddings) > 0 && len(sl.Embeddings[0]) != len(currentLayer.EmbeddingNeuron.Embeddings[0])) {
				return fmt.Errorf("mismatch in embedding matrix size at layer %d", i)
			}

			// Deep copy the 2D matrix
			currentLayer.EmbeddingNeuron.Embeddings = sl.Embeddings

		} else {
			// Case 2: Dense Layer
			if len(sl.Neurons) != len(currentLayer.Neurons) {
				return fmt.Errorf("mismatch: layer %d has %d saved neurons, but current network has %d",
					i, len(sl.Neurons), len(currentLayer.Neurons))
			}

			for j, snNeuron := range sl.Neurons {
				n := currentLayer.Neurons[j]

				if len(n.Weights) != len(snNeuron.Weights) {
					return fmt.Errorf("weight mismatch at layer %d neuron %d: expected %d weights, got %d",
						i, j, len(n.Weights), len(snNeuron.Weights))
				}

				// Safe to assign now
				copy(n.Weights, snNeuron.Weights)
				n.Bias = snNeuron.Bias
			}
		}
	}

	log.Println("Weights loaded successfully.")
	return nil
}

func (nw *Network) GetOutput() []float64 {
	numNeurons := len(nw.OutputLayer.InsToNext[0])

	outputs := make([]float64, numNeurons)
	for j := range numNeurons {
		outputs[j] = <-nw.OutputLayer.InsToNext[0][j]
	}
	return outputs
}

func (nw *Network) FeedForward(x []float64) {
	iLayer := nw.InputLayer
	m := len(iLayer.InsToNext)

	if m == 0 || len(iLayer.InsToNext[0]) == 0 {
		return
	}
	n := len(iLayer.InsToNext[0])

	for k := range m {
		for j := range n {
			iLayer.InsToNext[k][j] <- x[j]
		}
	}
}

func (nw *Network) Feedback(pred []float64, target []float64) {
	oLayer := nw.OutputLayer
	numOutputNeurons := len(oLayer.InsToNext[0])

	// Inject the final error into the output neurons to start backpropagation
	for i := range numOutputNeurons {
		oLayer.Neurons[i].ErrsFromNext[0] <- (pred[i] - target[i])
	}
}

func (nw *Network) WaitForBackpropFinish() {
	iLayer := nw.InputLayer

	// m is the number of rows/parallel inputs
	m := len(iLayer.ErrsFromNext)

	// n is the number of columns/features
	if m == 0 || len(iLayer.ErrsFromNext[0]) == 0 {
		panic("Should not happen in a valid network")
	}
	n := len(iLayer.ErrsFromNext[0])

	// Wait for the backpropagation signal on ALL m*n channels
	for k := range m {
		for j := range n {
			<-iLayer.ErrsFromNext[k][j]
		}
	}
}

func (nw *Network) Train(X [][]float64, Y []float64, cfg TrainingConfig) {
	log.Printf("Train: %+v", cfg)
	nw.UpdateNeuronCfg(cfg)

	start := time.Now()
	dataSize := len(X)

	limit := (dataSize / cfg.BatchSize) * cfg.BatchSize
	log.Printf("Dropped %d samples", dataSize-limit)

	for epoch := range cfg.Epochs {
		totalLoss := 0.0
		correct := 0
		X, Y = Shuffle(X, Y)
		for i := range limit {
			x := X[i]
			target := Y[i]

			// 1. Forward pass
			nw.FeedForward(x)
			pred := nw.GetOutput()

			// 2. Compute loss
			loss, predVector, targetVector := nw.computeLoss(pred, target, cfg)
			totalLoss += loss

			// 3. Backward pass
			nw.Feedback(predVector, targetVector)
			nw.WaitForBackpropFinish()

			// 4. Accuracy Check
			if nw.isCorrect(pred, target, cfg) {
				correct++
			}
		}
		// Logging
		if epoch%cfg.VerboseEvery == 0 {
			avgLoss := totalLoss / float64(dataSize)
			acc := float64(correct) / float64(dataSize) * 100.0
			elapsed := time.Since(start).Minutes()
			log.Printf("Epoch %d | Loss: %.6f | Accuracy: %.2f%% | Time: %.2f min\n", epoch, avgLoss, acc, elapsed)

			// if err := nw.SaveWeights("weights.json"); err != nil {
			// 	log.Println("Error saving weights:", err)
			// } else {
			// 	log.Println("Weights saved successfully.")
			// }
		}
	}
}

func (nw *Network) Predict(x []float64, cfg TrainingConfig) float64 {
	nw.FeedForward(x)
	predVector := nw.GetOutput()
	switch cfg.LossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		predVector = Softmax(predVector, 1.0)
		predictedClass := OneHotDecode(predVector)

		return predictedClass

	case BINARY_CROSS_ENTROPY:
		predScalar := sigmoid(predVector[0]) // Probability output

		// Determine the binary class based on the 0.5 threshold
		predictedClass := 0
		if predScalar >= 0.5 {
			predictedClass = 1
		}

		return float64(predictedClass)
	case MSE:
		predScalar := predVector[0]
		return predScalar

	default:
		// Handle uninitialized or unknown loss function
		log.Printf("Warning: Cannot log test result. Unknown loss function: %d\n", cfg.LossFunction)
	}
	return 0.0
}

func (nw *Network) PredictProbs(x []float64, cfg TrainingConfig) []float64 {
	// Forward pass
	nw.FeedForward(x)
	predVector := nw.GetOutput()

	switch cfg.LossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		return predVector // return logits; caller can apply Softmax with desired Temperature

	case BINARY_CROSS_ENTROPY:
		p := sigmoid(predVector[0])
		return []float64{1 - p, p} // return [P(class=0), P(class=1)]

	case MSE:
		return predVector // return regression output as is

	default:
		log.Printf("Warning: Unknown loss function in PredictProba: %d\n", cfg.LossFunction)
	}
	return []float64{}
}

func (nw *Network) Evaluate(X [][]float64, Y []float64, cfg TrainingConfig) (float64, float64) {
	totalLoss := 0.0
	correct := 0
	dataSize := len(X)

	for i := range X {
		x := X[i]
		target := Y[i]

		// 1. Forward pass
		nw.FeedForward(x)
		pred := nw.GetOutput()

		// 2. Compute loss
		loss, _, _ := nw.computeLoss(pred, target, cfg)
		totalLoss += loss

		// 3. Accuracy Check
		if nw.isCorrect(pred, target, cfg) {
			correct++
		}
	}
	avgLoss := totalLoss / float64(dataSize)
	acc := float64(correct) / float64(dataSize) * 100.0
	return avgLoss, acc
}

func (nw *Network) computeLoss(predVector []float64, targetScalar float64, cfg TrainingConfig) (float64, []float64, []float64) {
	switch cfg.LossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		// Ensure probabilities sum to 1
		predVector = Softmax(predVector, 1.0)

		// Convert scalar target to OHE
		targetVector := OneHotEncode(targetScalar, cfg.KClasses)

		loss := 0.0
		// L = - Sum[ y_i * log(y_hat_i) ]
		for k := range cfg.KClasses {
			y := targetVector[k]
			y_hat := predVector[k]

			// Add small epsilon to prevent log(0)
			epsilon := 1e-15
			y_hat_safe := math.Max(epsilon, y_hat)

			loss += -y * math.Log(y_hat_safe)
		}
		return loss, predVector, targetVector

	case BINARY_CROSS_ENTROPY:
		predScalar := sigmoid(predVector[0]) // y_hat

		// Add small epsilon to prevent log(0)
		epsilon := 1e-15

		// y * log(y_hat)
		term1 := targetScalar * math.Log(math.Max(epsilon, predScalar))

		// (1-y) * log(1 - y_hat)
		term2 := (1.0 - targetScalar) * math.Log(math.Max(epsilon, 1.0-predScalar))

		loss := -(term1 + term2)

		// Set targetVector
		return loss, []float64{predScalar}, []float64{targetScalar}

	case MSE:
		predScalar := predVector[0]
		return 0.5 * math.Pow(predScalar-targetScalar, 2), predVector, []float64{targetScalar}

	default:
		panic(fmt.Sprintf("Unknown loss function: %d", cfg.LossFunction))
	}
}

func (nw *Network) isCorrect(predVector []float64, targetScalar float64, cfg TrainingConfig) bool {
	switch cfg.LossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		predVector = Softmax(predVector, 1.0)

		predictedClass := OneHotDecode(predVector)

		// Compare predicted class index to the actual class index
		return float64(predictedClass) == targetScalar

	case BINARY_CROSS_ENTROPY:
		predScalar := sigmoid(predVector[0]) // The single probability output by the Sigmoid layer

		// The predicted class is 1 if the probability is >= 0.5, otherwise 0.
		predictedClass := 0
		if predScalar >= 0.5 {
			predictedClass = 1
		}

		// Compare predicted class (0 or 1) to the actual binary target (0 or 1)
		return float64(predictedClass) == targetScalar

	case MSE:
		// For regression, "accuracy" = % of predictions close to target
		predScalar := predVector[0]
		return math.Abs(predScalar-targetScalar) < 1
	}
	return false
}

func (nw *Network) UpdateNeuronCfg(cfg TrainingConfig) {
	for _, layer := range nw.Hidden {
		if layer.EmbeddingNeuron != nil {
			ack := make(chan struct{})
			layer.EmbeddingNeuron.ConfigUpdate <- NeuronCfg{
				LR:        cfg.LearningRate,
				BatchSize: cfg.BatchSize,
				Ack:       ack,
			}
			<-ack
		}

		for _, neuron := range layer.Neurons {
			ack := make(chan struct{})
			// Send with Ack channel
			neuron.ConfigUpdate <- NeuronCfg{
				LR:        cfg.LearningRate,
				BatchSize: cfg.BatchSize,
				Ack:       ack,
			}
			<-ack
		}
	}
}
