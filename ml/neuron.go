package ml

import (
	"log"
	"math"
	"math/rand"
	"slices"
	"sync"
)

const (
	leak = 0.01
)

var ChannelCapacity = 1

var activationMap = map[string]struct {
	fn ActivationFunc
	df ActivationFunc
}{
	"linear":    {linear, dfLinear},
	"sigmoid":   {sigmoid, dfSigmoid},
	"relu":      {relu, dfRelu},
	"leakyrelu": {leakyRelu, dfLeakyRelu},
	"tanh":      {tanh, dfTanh},
	"gelu":      {gelu, dfGelu},
}

type ActivationFunc func(float64) float64
type InitializerFunc func([]float64, int, int)

type Synapse struct {
	ID    int
	Value float64
}

type Neuron struct {
	ID      int
	Weights []float64
	Bias    float64
	LR      float64

	ErrsToPrev   []chan Synapse // send error backward to all neurons of previous layer connected to this neuron : write
	ErrsFromNext chan Synapse   // receive error from layer in front, from each neuron to which this neuron is connected to : read

	OutsFromPrev chan Synapse   // activated outputs from prev layer, outputs from all neurons of previos layer connected to this neuron  : read
	InsToNext    []chan Synapse // activated outputs to next layer, send output to all the neurons in the next layer to which this neuron is connected to : write

	BatchSize   int         // Stores the batch size
	BatchGradsW [][]float64 // To accumulate dL/dW for each sample in the batch
	BatchGradB  []float64   // To accumulate dL/dB for each sample in the batch
	BatchCount  int         // To track samples processed in current batch

	ConfigUpdate chan NeuronCfg // Channel to send new BatchSize, LR
}

type EmbeddingNeuron struct {
	Embeddings [][]float64 // vocabsize x dim
	LR         float64
	VocabSize  int // Stored for internal access in the goroutine

	ErrsToPrev   []chan Synapse // [InputLen] channels to send error backward to input layer
	ErrsFromNext chan Synapse   // [NextNeurons x OutputDim] channels to receive error from next layer
	OutsFromPrev chan Synapse   // [InputLen] channels to receive word indices
	InsToNext    []chan Synapse // [NextNeurons x OutputDim] channels to send flattened embeddings

	// Batching fields for accumulation
	BatchSize  int
	BatchCount int
	BatchGrads [][]float64 // VocabSize x EmbedDim (stores sum of gradients for batch)
	BatchUsage []int       // VocabSize (stores count of updates for each index)

	InputLen     int // InputLen to EmbeddingNeuron
	EmbedDim     int // Dimension per word (e.g., 2)
	ConfigUpdate chan NeuronCfg
}

type LSTMNeuron struct {
	ID int
	// Weights: 2D Array [4 Rows][InputSize + 1 Col]
	// Rows: 0=Forget, 1=Input, 2=Candidate, 3=Output
	// Cols: 0..M-1 are Input Weights, Last Col is Hidden Weight
	Weights [][]float64
	Biases  []float64 // [Bf, Bi, Bc, Bo]

	LR        float64 // Learning Rate
	ClipValue float64 // Gradient Clipping Value

	// Batching State
	BatchSize   int
	BatchCount  int
	BatchGradsW [][][]float64 // [BatchSize][4 Gates][InputSize+1]
	BatchGradsB [][]float64   // [BatchSize][4 Gates]

	// Communication Channels
	ErrsToPrev   []chan Synapse // Write to previous layer
	ErrsFromNext chan Synapse   // Read from next layer
	OutsFromPrev chan Synapse   // Read from previous layer
	InsToNext    []chan Synapse // Write to next layer

	// Signal to reset hidden state between sequences
	ResetChan    chan ResetState
	ConfigUpdate chan NeuronCfg
}

type NeuronCfg struct {
	LR        float64
	ClipValue float64
	BatchSize int
	Ack       chan struct{} // signal completion
}

// Information shared by forward to backward
type Knowledge struct {
	Input []float64
	T     float64
}

// LSTMState holds the snapshot of one timestep for Backpropagation
type LSTMState struct {
	X      []float64 // Inputs at this timestep
	H_prev float64   // Previous Hidden State
	C_prev float64   // Previous Cell State

	// Gate Activations (Post-activation values)
	F_gate float64
	I_gate float64
	C_cand float64
	O_gate float64

	C_curr float64 // Current Cell State
	H_curr float64 // Current Hidden State
}

type ResetState struct {
	Ack chan struct{} // signal completion
}

// Simple FIFO queue for neuron knowledge
type KnowledgeQueue []*Knowledge

func (q *KnowledgeQueue) Push(k *Knowledge) { *q = append(*q, k) }
func (q *KnowledgeQueue) Pop() *Knowledge {
	if len(*q) == 0 {
		return nil
	}
	k := (*q)[0]
	*q = (*q)[1:]
	return k
}

// Simple stack for LSTM History
type HistoryStack []*LSTMState

func (s *HistoryStack) Push(x *LSTMState) {
	*s = append(*s, x)
}

func (s *HistoryStack) Pop() *LSTMState {
	if len(*s) == 0 {
		return nil
	}
	last := len(*s) - 1
	v := (*s)[last]
	*s = (*s)[:last]
	return v
}

func NewNeuron(
	id int,
	errsToPrev []chan Synapse,
	outsFromPrev chan Synapse,
	errsFromNext chan Synapse,
	insToNext []chan Synapse,
	f, df ActivationFunc,
	init InitializerFunc,
) *Neuron {
	n := &Neuron{
		ID:           id,
		Weights:      make([]float64, len(errsToPrev)),
		Bias:         0.0,
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
		ConfigUpdate: make(chan NeuronCfg),
	}

	// Init weights
	init(n.Weights, len(n.ErrsToPrev), len(n.InsToNext))

	// Queue for knowledge
	queue := make(KnowledgeQueue, 0, ChannelCapacity) // Preallocate capacity so there is no memory reallocation

	// Global or Neuron-level Pool for Knowledge structs
	var knowledgePool = sync.Pool{
		New: func() any {
			return &Knowledge{}
		},
	}

	// Forward activation closure
	Activate := func(input []float64) float64 {
		t := n.Bias
		for i, val := range input {
			t += val * n.Weights[i]
		}

		kw := knowledgePool.Get().(*Knowledge)
		kw.Input = slices.Clone(input) // Pass a copy of the input slice
		kw.T = t

		// Push Knowledge to queue
		queue.Push(kw)
		return f(t)
	}

	go func() {
		// Forward pass
		inBuf := make([]float64, len(errsToPrev))
		inCounter := 0
		// Backward pass
		errFront := 0.0
		errCounter := 0
		// Gradient accumulators for batching
		avgBiasGrad := 0.0
		avgWeightsGrads := make([]float64, len(n.Weights))

		for {
			select {
			case cfg := <-n.ConfigUpdate:
				n.LR = cfg.LR
				n.BatchSize = cfg.BatchSize
				n.BatchCount = 0
				n.BatchGradsW = make([][]float64, cfg.BatchSize)
				n.BatchGradB = make([]float64, cfg.BatchSize)
				for i := range cfg.BatchSize {
					n.BatchGradsW[i] = make([]float64, len(n.Weights))
				}
				close(cfg.Ack)

			case in := <-n.OutsFromPrev:
				inBuf[in.ID] = in.Value
				inCounter++
				if inCounter == len(errsToPrev) {
					inCounter = 0
					out := Activate(inBuf)
					for i := range len(n.InsToNext) {
						n.InsToNext[i] <- Synapse{ID: n.ID, Value: out}
					}
				}

			case err := <-n.ErrsFromNext:
				errFront += err.Value
				errCounter++
				// wait for all incoming errors from front layer
				if errCounter == len(n.InsToNext) {
					// Retrieve Knowledge from queue
					kw := queue.Pop()
					grad := df(kw.T)

					// send error backward = w_current * grad(t) * errFront
					for i := range n.ErrsToPrev {
						n.ErrsToPrev[i] <- Synapse{ID: n.ID, Value: grad * n.Weights[i] * errFront}
					}

					if n.BatchSize == 1 {
						for i := range n.Weights {
							n.Weights[i] -= n.LR * grad * kw.Input[i] * errFront
						}
						n.Bias -= n.LR * grad * errFront

						// Return Knowledge struct to pool immediately
						kw.Input = nil
						kw.T = 0.0
						knowledgePool.Put(kw)

						errCounter = 0
						errFront = 0.0
						continue
					}

					// --- ACCUMULATION STEP (per sample) ---
					sampleIndex := n.BatchCount

					n.BatchGradB[sampleIndex] = grad * errFront
					for i := range n.Weights {
						n.BatchGradsW[sampleIndex][i] = grad * kw.Input[i] * errFront
					}

					n.BatchCount++
					// --- END ACCUMULATION STEP ---

					// --- BATCH UPDATE STEP ---
					if n.BatchCount == n.BatchSize {
						// 1. Calculate averages
						for i := range n.BatchSize {
							avgBiasGrad += n.BatchGradB[i]
							for j := range n.Weights {
								avgWeightsGrads[j] += n.BatchGradsW[i][j]
							}
						}
						avgBiasGrad /= float64(n.BatchSize)
						for j := range n.Weights {
							avgWeightsGrads[j] /= float64(n.BatchSize)
						}

						// 2. Update weights and bias (using the average gradients)
						for i := range n.Weights {
							n.Weights[i] -= n.LR * avgWeightsGrads[i]
						}
						n.Bias -= n.LR * avgBiasGrad

						// 3. Reset batch counter
						n.BatchCount = 0
						avgBiasGrad = 0.0
						for j := range n.Weights {
							avgWeightsGrads[j] = 0.0
						}
					}
					// --- END BATCH UPDATE STEP ---

					// Return Knowledge struct to pool
					kw.Input = nil // Clear reference to reusable input slice for safety
					kw.T = 0.0
					knowledgePool.Put(kw)

					errCounter = 0
					errFront = 0.0
				}
			}
		}
	}()

	return n
}

// NewEmbeddingNeuron creates the single Embedding block worker and its logic.
func NewEmbeddingNeuron(
	embedDim int,
	vocabSize int,
	inputLen int,
	errsToPrev []chan Synapse,
	errsFromNext chan Synapse,
	outsFromPrev chan Synapse,
	insToNext []chan Synapse,
	initializer InitializerFunc,
) *EmbeddingNeuron {
	en := &EmbeddingNeuron{
		Embeddings:   make([][]float64, vocabSize),
		VocabSize:    vocabSize, // Store for use in goroutine
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
		InputLen:     inputLen,
		EmbedDim:     embedDim,
		ConfigUpdate: make(chan NeuronCfg),
		BatchSize:    1, // Default to SGD
	}

	// Init embeddings: vocabSize x embedDim
	for i := range vocabSize {
		en.Embeddings[i] = make([]float64, embedDim)
		initializer(en.Embeddings[i], 0, 0) // Passing dummy in/out
	}

	// Initialize batching structures (for default BatchSize=1)
	// We use the full VocabSize as the accumulation buffer keys
	en.BatchGrads = make([][]float64, vocabSize)
	for i := range vocabSize {
		en.BatchGrads[i] = make([]float64, embedDim)
	}
	en.BatchUsage = make([]int, vocabSize)

	// Channel for internal forward/backward knowledge (transfers pointers for pooling)
	// ch := make(chan *Knowledge, ChannelCapacity)
	queue := make(KnowledgeQueue, 0, ChannelCapacity) // Preallocate capacity with some buffer

	// Global Pool for Knowledge structs (avoids high-frequency struct allocation)
	var knowledgePool = sync.Pool{
		New: func() any {
			return &Knowledge{}
		},
	}

	go func() {
		// GC Optimization: Allocate reusable buffers ONCE inside the goroutine scope
		inputIndices := make([]float64, en.InputLen)
		inCounter := 0

		outputDim := en.InputLen * en.EmbedDim
		embeddings := make([]float64, outputDim)

		// GC Optimization: Allocate reusable error buffer ONCE
		errFrontVec := make([]float64, outputDim) // Reusable buffer for aggregated errors
		errCounter := 0
		for {
			select {
			case cfg := <-en.ConfigUpdate:
				en.LR = cfg.LR
				en.BatchSize = cfg.BatchSize
				en.BatchCount = 0
				en.BatchGrads = make([][]float64, en.VocabSize)
				for i := range en.VocabSize {
					en.BatchGrads[i] = make([]float64, en.EmbedDim)
				}
				en.BatchUsage = make([]int, en.VocabSize)
				close(cfg.Ack)
			case in := <-en.OutsFromPrev:
				// 1. Read all input indices into the persistent buffer
				inputIndices[in.ID] = in.Value
				inCounter++
				if inCounter == en.InputLen {
					// 2. Get the embeddings and flatten
					for i, index := range inputIndices {
						idx := int(index)
						if idx < 0 || idx >= en.VocabSize {
							log.Printf("Warning: Word index %.4f out of bounds (0-%d)\n", index, en.VocabSize-1)
							continue
						}

						// Copy the embedding vector into the persistent output buffer
						srcVec := en.Embeddings[idx]
						copy(embeddings[i*en.EmbedDim:(i+1)*en.EmbedDim], srcVec)
					}

					// 3. Store indices for backward pass (using the pooled struct)
					kw := knowledgePool.Get().(*Knowledge)
					kw.Input = slices.Clone(inputIndices) // Pass a copy of the input indices

					queue.Push(kw)

					// 4. Send flattened outputs to next layer
					m := len(en.InsToNext)
					n := outputDim // len(en.InsToNext[0])

					for k := range m {
						for j := range n {
							en.InsToNext[k] <- Synapse{ID: j, Value: embeddings[j]}
						}
					}
					inCounter = 0
				}
			case err := <-en.ErrsFromNext:
				errFrontVec[err.ID] += err.Value
				errCounter++

				if errCounter == len(en.InsToNext)*outputDim {
					kw := queue.Pop()

					// 3. Send backpropagation finish signal (InputLen signals, e.g., 5)
					for i := range en.ErrsToPrev {
						en.ErrsToPrev[i] <- Synapse{ID: 1, Value: 1.0} // Signal finish
					}

					// --- SGD UPDATE (BatchSize == 1) ---
					if en.BatchSize == 1 {
						for i, index := range kw.Input {
							idx := int(index)
							if idx < 0 || idx >= en.VocabSize {
								continue
							}

							gradSlice := errFrontVec[i*en.EmbedDim : (i+1)*en.EmbedDim]
							// Update the embedding vector immediately (SGD)
							for j := range en.EmbedDim {
								en.Embeddings[idx][j] -= en.LR * gradSlice[j]
							}
						}

						// Return Knowledge struct to pool
						kw.Input = nil
						knowledgePool.Put(kw)

						for j := range outputDim {
							errFrontVec[j] = 0.0 // Zeroing the buffer
						}
						errCounter = 0
						continue
					}
					// --- END SGD UPDATE ---

					// --- ACCUMULATION STEP (for batch size > 1) ---
					for i, index := range kw.Input {
						idx := int(index)
						if idx < 0 || idx >= en.VocabSize {
							continue
						}

						gradSlice := errFrontVec[i*en.EmbedDim : (i+1)*en.EmbedDim]

						// Accumulate gradient sum and usage count
						for j := range en.EmbedDim {
							en.BatchGrads[idx][j] += gradSlice[j]
						}
						en.BatchUsage[idx]++
					}

					en.BatchCount++
					// --- END ACCUMULATION STEP ---

					// --- BATCH UPDATE STEP ---
					if en.BatchCount == en.BatchSize {
						// Apply the average gradient to all affected embeddings
						for idx := range en.VocabSize {
							if en.BatchUsage[idx] > 0 {
								// Update the embedding vector using the average gradient
								count := float64(en.BatchUsage[idx])
								for j := range en.EmbedDim {
									avgGrad := en.BatchGrads[idx][j] / count
									en.Embeddings[idx][j] -= en.LR * avgGrad
								}
							}

							// Reset accumulation for the next batch (Zero out existing buffers)
							for j := range en.EmbedDim {
								en.BatchGrads[idx][j] = 0.0
							}
							en.BatchUsage[idx] = 0
						}

						// Reset batch counter
						en.BatchCount = 0
					}
					// --- END BATCH UPDATE STEP ---

					// Return Knowledge struct to pool
					kw.Input = nil // Clear reference to reusable input slice for safety
					knowledgePool.Put(kw)

					for j := range outputDim {
						errFrontVec[j] = 0.0 // Zeroing the buffer
					}
					errCounter = 0
				}
			}
		}
	}()
	return en
}

func NewLSTMNeuron(
	id int,
	errsToPrev []chan Synapse,
	outsFromPrev chan Synapse,
	errsFromNext chan Synapse,
	insToNext []chan Synapse,
	timesteps int,
	initializer InitializerFunc,
) *LSTMNeuron {
	inputSize := len(errsToPrev)

	n := &LSTMNeuron{
		ID:           id,
		Weights:      make([][]float64, 4),
		Biases:       make([]float64, 4),
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
		ResetChan:    make(chan ResetState, 1),
		ConfigUpdate: make(chan NeuronCfg),
	}

	// Initialize Weights (4 Gates)
	for i := range 4 {
		n.Weights[i] = make([]float64, inputSize+1)
		initializer(n.Weights[i], len(n.ErrsToPrev), len(n.InsToNext))
		n.Biases[i] = 0.0
	}

	stack := make(HistoryStack, 0, ChannelCapacity*timesteps)

	// State Pool
	statePool := sync.Pool{
		New: func() any { return &LSTMState{} },
	}

	// --- Helper Math Functions ---
	sigmoid := func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
	dsigmoid := func(y float64) float64 { return y * (1.0 - y) }
	tanh := func(x float64) float64 { return math.Tanh(x) }
	dtanh := func(y float64) float64 { return 1.0 - (y * y) }

	// Gradient Clipping Helper
	clip := func(v, limit float64) float64 {
		if v > limit {
			return limit
		}
		if v < -limit {
			return -limit
		}
		return v
	}

	go func() {
		h_prev, c_prev := 0.0, 0.0
		numInputs := len(n.ErrsToPrev)
		inputs := make([]float64, numInputs)
		inCounter := 0

		incomingErrors := make([]float64, timesteps)
		timestep := 0
		errCounter := 0

		// Gradient Accumulators for Batch Averaging (reused memory)
		// These act as the temporary sum holders during the batch update step
		avgGradsW := make([][]float64, 4)
		for i := range 4 {
			avgGradsW[i] = make([]float64, numInputs+1)
		}
		avgGradsB := make([]float64, 4)

		calculateGate := func(inputs []float64, rowWeights []float64, h float64, bias float64) float64 {
			sum := bias
			for i, val := range inputs {
				sum += val * rowWeights[i]
			}
			sum += h * rowWeights[len(rowWeights)-1]
			return sum
		}

		for {
			select {
			case r := <-n.ResetChan:
				h_prev, c_prev = 0.0, 0.0
				stack = stack[:0] // Clear the stack history
				close(r.Ack)

			case cfg := <-n.ConfigUpdate:
				// --- CONFIG UPDATE STEP ---
				n.LR = cfg.LR
				n.ClipValue = cfg.ClipValue
				n.BatchSize = cfg.BatchSize
				n.BatchCount = 0

				// Preallocate Batch Accumulators
				// Dimensions: [BatchSize][4 Gates][InputSize + 1]
				n.BatchGradsW = make([][][]float64, cfg.BatchSize)
				n.BatchGradsB = make([][]float64, cfg.BatchSize)

				for i := range cfg.BatchSize {
					n.BatchGradsW[i] = make([][]float64, 4)
					n.BatchGradsB[i] = make([]float64, 4) // [4 Gates]
					for gate := range 4 {
						n.BatchGradsW[i][gate] = make([]float64, numInputs+1)
					}
				}
				close(cfg.Ack)

			case in := <-n.OutsFromPrev:
				inputs[in.ID] = in.Value
				inCounter++

				if inCounter == len(n.ErrsToPrev) {
					// 2. Prepare State Object
					st := statePool.Get().(*LSTMState)
					st.X = make([]float64, numInputs)
					copy(st.X, inputs)
					st.H_prev, st.C_prev = h_prev, c_prev

					// 3. Calculate Gates
					st.F_gate = sigmoid(calculateGate(inputs, n.Weights[0], h_prev, n.Biases[0]))
					st.I_gate = sigmoid(calculateGate(inputs, n.Weights[1], h_prev, n.Biases[1]))
					st.C_cand = tanh(calculateGate(inputs, n.Weights[2], h_prev, n.Biases[2]))
					st.O_gate = sigmoid(calculateGate(inputs, n.Weights[3], h_prev, n.Biases[3]))

					// 4. Update Memory
					st.C_curr = (st.F_gate * c_prev) + (st.I_gate * st.C_cand)
					st.H_curr = st.O_gate * tanh(st.C_curr)

					// 5. Store History
					stack.Push(st)

					// 6. Update Loop State
					h_prev, c_prev = st.H_curr, st.C_curr

					// 7. Send Output
					for _, ch := range n.InsToNext {
						ch <- Synapse{ID: n.ID, Value: st.H_curr}
					}
					inCounter = 0
				}

			case err := <-n.ErrsFromNext:
				incomingErrors[timestep] += err.Value
				errCounter++

				if errCounter == len(n.InsToNext) {
					timestep++

					if timestep < timesteps {
						errCounter = 0
						continue
					}

					// --- PHASE 2: BPTT CALCULATION ---
					dh_next, dC_next := 0.0, 0.0

					// Local Gradients for this sequence
					// Using explicit allocation here for clarity, relying on GC for these short-lived slices
					// or you could use a sync.Pool for these too if pressure is high.
					dW := make([][]float64, 4)
					for i := range dW {
						dW[i] = make([]float64, numInputs+1)
					}
					dB := make([]float64, 4)

					// Buffer for errors to send backward
					dx_buffer := make([][]float64, timesteps)

					historyCount := len(stack)

					// Iterate Backwards
					for t := historyCount - 1; t >= 0; t-- {
						st := stack.Pop()
						err_spatial := incomingErrors[t]

						dh_t := err_spatial + dh_next

						tanh_C := tanh(st.C_curr)
						d_o := dh_t * tanh_C * dsigmoid(st.O_gate)
						dC_t := (dh_t * st.O_gate * dtanh(tanh_C)) + dC_next

						d_c_cand := dC_t * st.I_gate * dtanh(st.C_cand)
						d_i := dC_t * st.C_cand * dsigmoid(st.I_gate)
						d_f := dC_t * st.C_prev * dsigmoid(st.F_gate)

						deltas := []float64{d_f, d_i, d_c_cand, d_o}

						for _, d := range deltas {
							if math.IsNaN(d) || math.IsInf(d, 0) {
								panic("NaN Gradient in LSTM")
							}
						}

						// Accumulate Gradients for this sample
						for gateIdx := range 4 {
							d_gate := deltas[gateIdx]
							for j := range numInputs {
								dW[gateIdx][j] += d_gate * st.X[j]
							}
							dW[gateIdx][numInputs] += d_gate * st.H_prev
							dB[gateIdx] += d_gate
						}

						// Calculate dx for previous layer
						dx_buffer[t] = make([]float64, numInputs)
						for j := range numInputs {
							dx_j := (d_f * n.Weights[0][j]) +
								(d_i * n.Weights[1][j]) +
								(d_c_cand * n.Weights[2][j]) +
								(d_o * n.Weights[3][j])

							dx_buffer[t][j] = dx_j
						}

						// Temporal Error
						lastCol := numInputs
						dh_prev_step := (d_f * n.Weights[0][lastCol]) +
							(d_i * n.Weights[1][lastCol]) +
							(d_c_cand * n.Weights[2][lastCol]) +
							(d_o * n.Weights[3][lastCol])

						dh_next = dh_prev_step
						dC_next = dC_t * st.F_gate

						st.X = nil
						statePool.Put(st)
					}

					// --- PHASE 3: ERROR PROPAGATION (CHRONOLOGICAL) ---
					for t := range historyCount {
						for j := range numInputs {
							n.ErrsToPrev[j] <- Synapse{ID: n.ID, Value: dx_buffer[t][j]}
						}
					}

					// --- PHASE 4: UPDATE STRATEGY (BATCHING) ---

					// Immediate Update (BatchSize == 1)
					if n.BatchSize == 1 {
						for r := range 4 {
							for c := range numInputs + 1 {
								// Clip and Update
								n.Weights[r][c] -= n.LR * clip(dW[r][c], n.ClipValue)
							}
							n.Biases[r] -= n.LR * clip(dB[r], n.ClipValue)
						}
					} else {
						// Batch Accumulation
						idx := n.BatchCount
						for r := range 4 {
							copy(n.BatchGradsW[idx][r], dW[r])
							n.BatchGradsB[idx][r] = dB[r]
						}
						n.BatchCount++

						// Batch Update Trigger
						if n.BatchCount == n.BatchSize {
							// 1. Reset Averages
							for r := range 4 {
								avgGradsB[r] = 0.0
								for c := range numInputs + 1 {
									avgGradsW[r][c] = 0.0
								}
							}

							// 2. Sum Gradients
							for b := range n.BatchSize {
								for r := range 4 {
									avgGradsB[r] += n.BatchGradsB[b][r]
									for c := range numInputs + 1 {
										avgGradsW[r][c] += n.BatchGradsW[b][r][c]
									}
								}
							}

							// 3. Average & Apply
							batchFloat := float64(n.BatchSize)
							for r := range 4 {
								// Average Bias
								avgB := avgGradsB[r] / batchFloat
								n.Biases[r] -= n.LR * clip(avgB, n.ClipValue)

								for c := range numInputs + 1 {
									// Average Weight
									avgW := avgGradsW[r][c] / batchFloat
									n.Weights[r][c] -= n.LR * clip(avgW, n.ClipValue)
								}
							}

							// 4. Reset Batch Count
							n.BatchCount = 0
						}
					}

					// --- CLEANUP ---
					for j := range timesteps {
						incomingErrors[j] = 0.0
					}
					errCounter = 0
					timestep = 0
				}
			}
		}
	}()
	return n
}

func (n *LSTMNeuron) ResetState() {
	ack := make(chan struct{})
	n.ResetChan <- ResetState{Ack: ack}
	<-ack
}

// Activations and Derivatives
func linear(x float64) float64 {
	return x
}

func dfLinear(x float64) float64 {
	return 1
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dfSigmoid(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func dfRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func leakyRelu(x float64) float64 {
	if x > 0 {
		return x
	}
	return leak * x
}

func dfLeakyRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return leak
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func dfTanh(x float64) float64 {
	t := math.Tanh(x)
	return 1 - t*t
}

func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

func dfGelu(x float64) float64 {
	// Approx derivative of GELU
	const c = 0.0356774
	const b = 0.797885
	t := math.Tanh(b * (x + c*math.Pow(x, 3)))
	return 0.5 * (1 + t + x*(1-t*t)*(b+3*b*c*x*x))
}

// Weight Initializers
func XavierInit(weights []float64, fanIn, fanOut int) {
	limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
	for i := range weights {
		weights[i] = rand.Float64()*(2*limit) - limit
	}
}

func XavierNormal(weights []float64, fanIn, fanOut int) {
	std := math.Sqrt(2.0 / float64(fanIn+fanOut))
	for i := range weights {
		weights[i] = rand.NormFloat64() * std
	}
}

func HeNormal(weights []float64, fanIn, fanOut int) {
	std := math.Sqrt(2.0 / float64(fanIn))
	for i := range weights {
		weights[i] = rand.NormFloat64() * std
	}
}

func HeUniform(weights []float64, fanIn, fanOut int) {
	limit := math.Sqrt(6.0 / float64(fanIn))
	for i := range weights {
		weights[i] = rand.Float64()*(2*limit) - limit
	}
}

func Random(weights []float64, fanIn, fanOut int) {
	for i := range weights {
		weights[i] = rand.Float64()*0.2 - 0.1
	}
}
