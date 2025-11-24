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
	Epochs       int     // Number of training epochs
	BatchSize    int     // Number of samples per batch
	LearningRate float64 // Learning rate for weight updates
	ClipValue    float64 // For gradient clipping
	LossFunction int     // MSE, CATEGORICAL_CROSS_ENTROPY, BINARY_CROSS_ENTROPY
	KClasses     int     // For CATEGORICAL_CROSS_ENTROPY
	VerboseEvery int     // How often to log progress (in epochs)
	ShuffleData  bool    // Whether to shuffle data each epoch
	MinAccToSave float64 // Minimum accuracy threshold to save weights
}

type SerializableNeuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type SerializableLSTM struct {
	Weights [][]float64 `json:"weights"`
	Biases  []float64   `json:"biases"`
}

type SerializableLayer struct {
	Neurons    []SerializableNeuron `json:"neurons,omitempty"`
	Embeddings [][]float64          `json:"embeddings,omitempty"`
	LSTMs      []SerializableLSTM   `json:"lstms,omitempty"`
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

	if def0.Type == LayerTypeEmbedding {
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
		panic("First Dense must be called with two args: Dense(m, inputDim)")

	} else {
		// Only Dense Input Setup: m=def0.Neurons, n=def0.InputNeurons
		iLayer = NewInputLayer(def0.Neurons, def0.InputNeurons)
		prevErrs = iLayer.ErrsFromNext
		prevIns = iLayer.InsToNext
		loopStart = 0 // Start loop from defs[0]
	}

	// 2. Create subsequent Dense layers
	// Start from the next definition (defs[loopStart]) up to the second-to-last layer.
	for i := loopStart; i < len(defs)-1; i++ {
		l := NewHiddenLayer(prevErrs, prevIns, defs[i], defs[i+1])
		hidden = append(hidden, l)

		// advance the prevs to this layer's outputs for the next iteration
		prevErrs = l.ErrsFromNext
		prevIns = l.InsToNext
	}

	// 3. Last layer (using the last definition in the slice)
	l := NewOutputLayer(prevErrs, prevIns, defs[len(defs)-1])
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

		switch {
		case layer.EmbeddingNeuron != nil:
			// Case 1: Embedding Layer
			sl.Embeddings = layer.EmbeddingNeuron.Embeddings

		case len(layer.LSTMNeurons) > 0:
			// Case 2: LSTM Layer
			for _, lstm := range layer.LSTMNeurons {
				sl.LSTMs = append(sl.LSTMs, SerializableLSTM{
					Weights: lstm.Weights,
					Biases:  lstm.Biases,
				})
			}

		default:
			// Case 3: Dense Layer
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
		return fmt.Errorf("mismatch: saved network has %d layers, current network has %d",
			len(sn.Layers), len(nw.Hidden))
	}

	for i, sl := range sn.Layers {
		currentLayer := nw.Hidden[i]

		switch {
		case currentLayer.EmbeddingNeuron != nil:
			// ===== Embedding =====
			if len(sl.Embeddings) == 0 {
				return fmt.Errorf("layer %d expected embeddings but none found", i)
			}
			currentLayer.EmbeddingNeuron.Embeddings = sl.Embeddings

		case len(currentLayer.LSTMNeurons) > 0:
			// ===== LSTM =====
			if len(sl.LSTMs) != len(currentLayer.LSTMNeurons) {
				return fmt.Errorf("layer %d mismatch lstm count: saved=%d current=%d",
					i, len(sl.LSTMs), len(currentLayer.LSTMNeurons))
			}

			for j, saved := range sl.LSTMs {
				lstm := currentLayer.LSTMNeurons[j]

				// Check size
				if len(saved.Weights) != len(lstm.Weights) ||
					len(saved.Weights[0]) != len(lstm.Weights[0]) {
					return fmt.Errorf("layer %d lstm %d weight size mismatch", i, j)
				}

				// Deep copy
				for r := range lstm.Weights {
					copy(lstm.Weights[r], saved.Weights[r])
				}
				copy(lstm.Biases, saved.Biases)
			}

		default:
			// ===== Dense =====
			if len(sl.Neurons) != len(currentLayer.Neurons) {
				return fmt.Errorf("mismatch: layer %d wrong neurons count", i)
			}
			for j, snNeuron := range sl.Neurons {
				n := currentLayer.Neurons[j]
				copy(n.Weights, snNeuron.Weights)
				n.Bias = snNeuron.Bias
			}
		}
	}

	log.Println("Weights loaded successfully.")
	return nil
}

func (nw *Network) LogAndSaveWeights(epoch int, totalLoss float64, correct int, limit int, start time.Time, cfg TrainingConfig) {
	if epoch%cfg.VerboseEvery == 0 {
		// Calculate metrics based on processed samples (limit)
		avgLoss := totalLoss / float64(limit)
		acc := float64(correct) / float64(limit) * 100.0
		elapsed := time.Since(start).Minutes()

		log.Printf("Epoch %d | Loss: %.6f | Accuracy: %.2f%% | Time: %.2f min\n", epoch, avgLoss, acc, elapsed)

		if acc > cfg.MinAccToSave {
			filename := fmt.Sprintf("assets/weights_epoch_%d_acc_%.2f.json", epoch, acc)
			if err := nw.SaveWeights(filename); err != nil {
				log.Println("Error saving weights:", err)
			} else {
				log.Println("Weights saved successfully.")
			}
		}
	}
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

func (nw *Network) ResetLSTMState() {
	for _, layer := range nw.Hidden {
		for _, neuron := range layer.LSTMNeurons {
			neuron.ResetState()
		}
	}
}

func (nw *Network) Train(X [][]float64, Y any, cfg TrainingConfig) {
	// Log and Configuration Setup (Common to both)
	log.Printf("Train: %+v", cfg)
	nw.UpdateNeuronCfg(cfg)

	start := time.Now()
	dataSize := len(X)

	// Calculate the limit for batch processing (Common to both)
	limit := (dataSize / cfg.BatchSize) * cfg.BatchSize
	if dataSize-limit > 0 {
		log.Printf("Dropped %d samples (not enough for a full batch)", dataSize-limit)
	}

	// --- Core Logic Dispatcher (Type check only happens ONCE) ---
	switch targets := Y.(type) {
	case []float64:
		nw.trainNonSequential(X, targets, cfg, limit, dataSize, start)
	case [][]float64:
		nw.trainSequential(X, targets, cfg, limit, dataSize, start)
	default:
		log.Fatalf("Unsupported target data type for Y: %T. Must be []float64 (non-sequential) or [][]float64 (sequential).", Y)
	}
}

// --- Internal Training Functions ---
// trainNonSequential handles standard, non-RNN training (Y is []float64).
func (nw *Network) trainNonSequential(X [][]float64, Y []float64, cfg TrainingConfig, limit, dataSize int, start time.Time) {
	for epoch := range cfg.Epochs {
		totalLoss := 0.0
		correct := 0

		if cfg.ShuffleData {
			Shuffle(X, Y)
		}

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

		// Logging and Saving Weights
		nw.LogAndSaveWeights(epoch, totalLoss, correct, limit, start, cfg)
	}
}

// trainSequential handles sequence (RNN/LSTM) training (Y is [][]float64).
func (nw *Network) trainSequential(X [][]float64, Y [][]float64, cfg TrainingConfig, limit, dataSize int, start time.Time) {
	if dataSize == 0 || len(X[0]) == 0 {
		log.Println("Sequential training requires non-empty data with timesteps.")
		return
	}
	if cfg.BatchSize != 1 {
		panic("Sequential training currently supports only BatchSize=1.")
	}
	timesteps := len(X[0])

	// These slices store data across the sequence for backprop-through-time
	predVecs := make([][]float64, timesteps)
	targetVecs := make([][]float64, timesteps)

	for epoch := range cfg.Epochs {
		totalLoss := 0.0
		correct := 0

		if cfg.ShuffleData {
			Shuffle(X, Y)
		}

		for i := range limit {
			nw.ResetLSTMState()

			Xseq := X[i]
			Yseq := Y[i]

			var lastPred []float64
			var lastTarget float64

			// Forward Pass (through time)
			for t := range timesteps {
				// We assume Xseq is a flattened sequence, so we take a single element for input
				x_t := []float64{Xseq[t]}
				target_t := Yseq[t]

				// 1. Forward pass for timestep t
				nw.FeedForward(x_t)
				pred_t := nw.GetOutput()

				// 2. Compute loss for timestep t
				loss, predVector, targetVector := nw.computeLoss(pred_t, target_t, cfg)
				totalLoss += loss

				predVecs[t] = predVector
				targetVecs[t] = targetVector

				// Keep track of the last step for accuracy check
				lastPred = pred_t
				lastTarget = target_t
			}

			// 3. Backward Pass (through time)
			for t := range timesteps {
				nw.Feedback(predVecs[t], targetVecs[t])
			}
			// Wait for all backpropagation to finish
			for range timesteps {
				nw.WaitForBackpropFinish()
			}

			// 4. Accuracy Check (only on the final output of the sequence)
			if nw.isCorrect(lastPred, lastTarget, cfg) {
				correct++
			}
		}

		// Logging and Saving Weights
		nw.LogAndSaveWeights(epoch, totalLoss, correct, limit, start, cfg)
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

func (nw *Network) PredictSeq(x []float64, cfg TrainingConfig) float64 {
	nw.ResetLSTMState()

	for t := range x {
		nw.FeedForward([]float64{x[t]})
	}
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

func (nw *Network) EvaluateSeq(X [][]float64, Y []float64, cfg TrainingConfig) (float64, float64) {
	totalLoss := 0.0
	correct := 0
	dataSize := len(X)

	for i := range X {
		nw.ResetLSTMState()

		seq := X[i] // full sequence
		target := Y[i]

		// run through timesteps
		for t := range seq {
			nw.FeedForward([]float64{seq[t]})
		}

		pred := nw.GetOutput()

		loss, _, _ := nw.computeLoss(pred, target, cfg)
		totalLoss += loss

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

		for _, neuron := range layer.LSTMNeurons {
			ack := make(chan struct{})
			// Send with Ack channel
			neuron.ConfigUpdate <- NeuronCfg{
				LR:        cfg.LearningRate,
				ClipValue: cfg.ClipValue,
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
