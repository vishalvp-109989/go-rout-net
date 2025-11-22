package ml

import (
	"fmt"
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

type Neuron struct {
	Weights []float64
	Bias    float64
	LR      float64

	ErrsToPrev   []chan float64 // send error backward to all neurons of previous layer connected to this neuron : write
	ErrsFromNext []chan float64 // receive error from layer in front, from each neuron to which this neuron is connected to : read

	OutsFromPrev []chan float64 // activated outputs from prev layer, outputs from all neurons of previos layer connected to this neuron  : read
	InsToNext    []chan float64 // activated outputs to next layer, send output to all the neurons in the next layer to which this neuron is connected to : write

	BatchSize   int         // Stores the batch size
	BatchGradsW [][]float64 // To accumulate dL/dW for each sample in the batch
	BatchGradB  []float64   // To accumulate dL/dB for each sample in the batch
	BatchCount  int         // To track samples processed in current batch

	// A channel to safely signal config changes to the backward worker
	ConfigUpdate chan NeuronCfg // Channel to send new BatchSize
	mu           sync.RWMutex   // protects Weights, Biases, History, HistoryCount
}

type EmbeddingNeuron struct {
	Embeddings [][]float64 // vocabsize x dim
	LR         float64
	VocabSize  int // Stored for internal access in the goroutine

	ErrsToPrev   []chan float64   // [InputLen] channels to send error backward to input layer
	ErrsFromNext [][]chan float64 // [NextNeurons x OutputDim] channels to receive error from next layer
	OutsFromPrev []chan float64   // [InputLen] channels to receive word indices
	InsToNext    [][]chan float64 // [NextNeurons x OutputDim] channels to send flattened embeddings

	// Batching fields for accumulation
	BatchSize  int
	BatchCount int
	BatchGrads [][]float64 // VocabSize x EmbedDim (stores sum of gradients for batch)
	BatchUsage []int       // VocabSize (stores count of updates for each index)

	InputLen     int // InputLen to EmbeddingNeuron
	EmbedDim     int // Dimension per word (e.g., 2)
	ConfigUpdate chan NeuronCfg
	mu           sync.RWMutex // protects Weights, Biases, History, HistoryCount
}

type LSTMNeuron struct {
	// Weights: 2D Array [4 Rows][InputSize + 1 Col]
	// Rows: 0=Forget, 1=Input, 2=Candidate, 3=Output
	// Cols: 0..M-1 are Input Weights, Last Col is Hidden Weight
	Weights [][]float64
	Biases  []float64 // [Bf, Bi, Bc, Bo]

	LR        float64 // Learning Rate
	ClipValue float64 // Gradient Clipping Value

	// Communication Channels
	ErrsToPrev   []chan float64 // Write to previous layer
	ErrsFromNext []chan float64 // Read from next layer
	OutsFromPrev []chan float64 // Read from previous layer
	InsToNext    []chan float64 // Write to next layer

	// Signal to reset hidden state between sequences
	ResetChan chan ResetState

	// Internal Memory for the current sequence
	History      []*LSTMState
	HistoryCount int

	ConfigUpdate chan NeuronCfg
	mu           sync.RWMutex
}

type NeuronCfg struct {
	LR        float64
	ClipValue float64
	BatchSize int
	Ack       chan struct{} // signal completion
}

// Information stored between forward and backward
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

func NewNeuron(errsToPrev, outsFromPrev, errsFromNext, insToNext []chan float64, f, df ActivationFunc, init InitializerFunc) *Neuron {
	n := &Neuron{
		Weights:      make([]float64, len(outsFromPrev)),
		Bias:         0.0,
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
		ConfigUpdate: make(chan NeuronCfg),
		mu:           sync.RWMutex{},
	}

	// Init weights
	init(n.Weights, len(n.OutsFromPrev), len(n.InsToNext))

	// Channel for internal forward/backward knowledge
	ch := make(chan *Knowledge, ChannelCapacity)

	// Global or Neuron-level Pool for Knowledge structs
	var knowledgePool = sync.Pool{
		New: func() any {
			return &Knowledge{}
		},
	}

	// Forward activation closure
	Activate := func(input []float64) float64 {
		n.mu.Lock()
		t := n.Bias
		for i, val := range input {
			t += val * n.Weights[i]
		}
		n.mu.Unlock()

		kw := knowledgePool.Get().(*Knowledge)
		kw.Input = slices.Clone(input) // Set the reference to the input slice
		kw.T = t

		// Store forward knowledge for backward use
		select {
		case ch <- kw: // Send the value (or pointer if the channel is for pointers)
		default:
			// log.Println("Failed to store knowledge")
			knowledgePool.Put(kw) // Return to pool if not used
		}
		return f(t)
	}

	// Launch forward worker goroutine
	go func() {
		in := make([]float64, len(outsFromPrev))
		for {
			for i := range n.OutsFromPrev {
				in[i] = <-n.OutsFromPrev[i]
			}
			out := Activate(in)
			for i := range len(n.InsToNext) {
				n.InsToNext[i] <- out
			}
		}
	}()

	// Launch backward worker goroutine
	go func() {
		// --- CONFIG UPDATE STEP ---
		cfg := <-n.ConfigUpdate
		n.mu.Lock()
		n.LR = cfg.LR
		n.BatchSize = cfg.BatchSize
		n.BatchCount = 0 // Reset counter
		n.BatchGradsW = make([][]float64, cfg.BatchSize)
		n.BatchGradB = make([]float64, cfg.BatchSize)
		for i := range cfg.BatchSize {
			n.BatchGradsW[i] = make([]float64, len(n.Weights))
		}
		n.mu.Unlock()
		close(cfg.Ack)

		// avgWeightsGrads and avgBiasGrad are moved here and are only allocated once
		// when the goroutine starts, reducing their scope/lifetime issues.
		avgBiasGrad := 0.0
		avgWeightsGrads := make([]float64, len(n.Weights))

		for {
			// wait for forward pass to store state
			kw := <-ch
			// compute local gradient
			grad := df(kw.T)

			// wait for all incoming errors from front layer
			errFront := 0.0
			for _, outCh := range n.ErrsFromNext {
				errFront += <-outCh
			}

			// send error backward = w_current * grad(t) * errFront
			for i := range n.ErrsToPrev {
				n.ErrsToPrev[i] <- grad * n.Weights[i] * errFront
			}

			if n.BatchSize == 1 {
				n.mu.Lock()
				for i := range n.Weights {
					n.Weights[i] -= n.LR * grad * kw.Input[i] * errFront
				}
				n.Bias -= n.LR * grad * errFront
				n.mu.Unlock()
				// Return Knowledge struct to pool immediately
				kw.Input = nil
				kw.T = 0.0
				knowledgePool.Put(kw)
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
				n.mu.Lock()
				// 2. Update weights and bias (using the average gradients)
				for i := range n.Weights {
					n.Weights[i] -= n.LR * avgWeightsGrads[i]
				}
				n.Bias -= n.LR * avgBiasGrad
				n.mu.Unlock()
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

		}
	}()

	return n
}

// NewEmbeddingNeuron creates the single Embedding block worker and its logic.
// Renamed from user's request `NewEmbeddingNeuron` to return `*Embedding`.
func NewEmbeddingNeuron(
	embedDim int,
	vocabSize int,
	inputLen int,
	errsToPrev []chan float64,
	errsFromNext [][]chan float64,
	outsFromPrev []chan float64,
	insToNext [][]chan float64,
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
		mu:           sync.RWMutex{},
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
	ch := make(chan *Knowledge, ChannelCapacity)

	// Global Pool for Knowledge structs (avoids high-frequency struct allocation)
	var knowledgePool = sync.Pool{
		New: func() any {
			return &Knowledge{}
		},
	}

	// Launch forward worker goroutine
	go func() {
		// GC Optimization: Allocate reusable buffers ONCE inside the goroutine scope
		inputIndices := make([]float64, en.InputLen)
		outputDim := en.InputLen * en.EmbedDim
		embeddings := make([]float64, outputDim)

		for {
			// 1. Read all input indices into the persistent buffer
			for i := range en.InputLen {
				inputIndices[i] = <-en.OutsFromPrev[i]
			}
			en.mu.Lock()
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
			en.mu.Unlock()

			// 3. Store indices for backward pass (using the pooled struct)
			kw := knowledgePool.Get().(*Knowledge)
			kw.Input = slices.Clone(inputIndices) //inputIndices // Pass the reference to the reusable buffer

			select {
			case ch <- kw:
				// Stored successfully
			default:
				// log.Println("Failed to store knowledge in EmbeddingNeuron forward pass")
				knowledgePool.Put(kw) // Return to pool if send fails
			}

			// 4. Send flattened outputs to next layer
			m := len(en.InsToNext)
			n := len(en.InsToNext[0])

			for k := range m {
				for j := range n {
					en.InsToNext[k][j] <- embeddings[j]
				}
			}
		}
	}()

	// Launch backward worker goroutine
	go func() {
		// --- CONFIG UPDATE STEP ---
		cfg := <-en.ConfigUpdate
		en.mu.Lock()
		en.LR = cfg.LR
		en.BatchSize = cfg.BatchSize
		en.BatchCount = 0
		en.BatchGrads = make([][]float64, en.VocabSize)
		for i := range en.VocabSize {
			en.BatchGrads[i] = make([]float64, en.EmbedDim)
		}
		en.BatchUsage = make([]int, en.VocabSize)
		en.mu.Unlock()
		close(cfg.Ack)

		// GC Optimization: Allocate reusable error buffer ONCE
		m := len(en.ErrsFromNext)
		n := len(en.ErrsFromNext[0])
		errFrontVec := make([]float64, n) // Reusable buffer for aggregated errors

		for {
			// wait for forward pass to store state
			kw := <-ch
			// 1. wait for all incoming errors from front layer and aggregate into reusable buffer
			// IMPORTANT: Reset the reusable buffer before aggregation
			for j := range n {
				errFrontVec[j] = 0.0
			}

			// Aggregate errors for each feature (j) from all next layer's neurons (k)
			for k := range m {
				for j := range n {
					errFrontVec[j] += <-en.ErrsFromNext[k][j]
				}
			}

			// 3. Send backpropagation finish signal (InputLen signals, e.g., 5)
			for i := range en.ErrsToPrev {
				en.ErrsToPrev[i] <- 1.0 // Signal finish
			}

			// --- SGD UPDATE (BatchSize == 1) ---
			if en.BatchSize == 1 {
				for i, index := range kw.Input {
					idx := int(index)
					if idx < 0 || idx >= en.VocabSize {
						continue
					}

					gradSlice := errFrontVec[i*en.EmbedDim : (i+1)*en.EmbedDim]
					en.mu.Lock()
					// Update the embedding vector immediately (SGD)
					for j := range en.EmbedDim {
						en.Embeddings[idx][j] -= en.LR * gradSlice[j]
					}
					en.mu.Unlock()
				}

				// Return Knowledge struct to pool
				kw.Input = nil
				knowledgePool.Put(kw)
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
						en.mu.Lock()
						for j := range en.EmbedDim {
							avgGrad := en.BatchGrads[idx][j] / count
							en.Embeddings[idx][j] -= en.LR * avgGrad
						}
						en.mu.Unlock()
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

		}
	}()

	return en
}

func NewLSTMNeuron(errsToPrev, outsFromPrev, errsFromNext, insToNext []chan float64, timesteps int, initializer InitializerFunc) *LSTMNeuron {
	inputSize := len(outsFromPrev)

	n := &LSTMNeuron{
		Weights:      make([][]float64, 4),
		Biases:       make([]float64, 4),
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
		ResetChan:    make(chan ResetState, 1),
		History:      make([]*LSTMState, timesteps),
		ConfigUpdate: make(chan NeuronCfg),
		mu:           sync.RWMutex{},
	}

	// Initialize Weights (4 Gates)
	for i := range 4 {
		n.Weights[i] = make([]float64, inputSize+1)
		initializer(n.Weights[i], len(n.OutsFromPrev), len(n.InsToNext))
		n.Biases[i] = 0.0
	}

	// State Pool
	statePool := sync.Pool{
		New: func() any { return &LSTMState{} },
	}

	// --- Helper Math Functions ---
	sigmoid := func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
	dsigmoid := func(y float64) float64 { return y * (1.0 - y) }
	tanh := func(x float64) float64 { return math.Tanh(x) }
	dtanh := func(y float64) float64 { return 1.0 - (y * y) }

	// --- NEW: Gradient Clipping Helper ---
	clip := func(v, limit float64) float64 {
		if v > limit {
			return limit
		}
		if v < -limit {
			return -limit
		}
		return v
	}

	// ---------------------------------------------------------
	// FORWARD GOROUTINE
	// ---------------------------------------------------------
	go func() {
		h_prev, c_prev := 0.0, 0.0
		numInputs := len(n.OutsFromPrev)

		calculateGate := func(inputs []float64, rowWeights []float64, h float64, bias float64) float64 {
			sum := bias
			for i, val := range inputs {
				sum += val * rowWeights[i]
			}
			sum += h * rowWeights[len(rowWeights)-1]
			return sum
		}

		for {
		READ_INPUTS:
			// 1. Collect Inputs (Blocks here)
			inputs := make([]float64, numInputs)
			for i, ch := range n.OutsFromPrev {
				select {
				case r := <-n.ResetChan:
					h_prev, c_prev = 0.0, 0.0
					n.mu.Lock()
					n.HistoryCount = 0
					n.mu.Unlock()
					close(r.Ack)
					goto READ_INPUTS
				case inputs[i] = <-ch:
				}
			}

			// 2. Prepare State Object
			st := statePool.Get().(*LSTMState)
			st.X = make([]float64, numInputs)
			copy(st.X, inputs) // Copy is crucial!
			st.H_prev, st.C_prev = h_prev, c_prev

			// 3. Calculate Gates
			// Snapshot weights & biases FIRST
			n.mu.RLock()
			wcopy := make([][]float64, 4)
			for i := range 4 {
				wcopy[i] = slices.Clone(n.Weights[i])
			}
			bcopy := slices.Clone(n.Biases)
			n.mu.RUnlock()

			// Now compute gates using snapshot
			st.F_gate = sigmoid(calculateGate(inputs, wcopy[0], h_prev, bcopy[0]))
			st.I_gate = sigmoid(calculateGate(inputs, wcopy[1], h_prev, bcopy[1]))
			st.C_cand = tanh(calculateGate(inputs, wcopy[2], h_prev, bcopy[2]))
			st.O_gate = sigmoid(calculateGate(inputs, wcopy[3], h_prev, bcopy[3]))

			// 4. Update Memory
			st.C_curr = (st.F_gate * c_prev) + (st.I_gate * st.C_cand)
			st.H_curr = st.O_gate * tanh(st.C_curr)

			// 5. Store History
			// n.History = append(n.History, st)
			n.mu.Lock()
			n.History[n.HistoryCount] = st
			n.HistoryCount++
			n.mu.Unlock()

			// 6. Update Loop State
			h_prev, c_prev = st.H_curr, st.C_curr

			// 7. Send Output
			for _, ch := range n.InsToNext {
				ch <- st.H_curr
			}
		}
	}()

	// ---------------------------------------------------------
	// BACKWARD GOROUTINE
	// ---------------------------------------------------------
	go func() {
		cfg := <-n.ConfigUpdate
		// --- CONFIG UPDATE STEP ---
		n.LR = cfg.LR
		n.ClipValue = cfg.ClipValue // Clipping Threshold (Standard values are 1.0 or 5.0)
		close(cfg.Ack)

		numInputs := len(n.OutsFromPrev)
		for {
			// --- PHASE 1: ERROR COLLECTION ---
			incomingErrors := make([]float64, timesteps)
			// Collect remaining errors (Assumes Order 0 -> N)
			for t := range timesteps {
				sumErr := 0.0
				for _, ch := range n.ErrsFromNext {
					val := <-ch
					if math.IsNaN(val) || math.IsInf(val, 0) {
						panic(fmt.Sprintf("LSTM Received NaN error from Output Layer at timestep %d", t))
					}
					// sumErr += <-ch
					sumErr += val
				}
				incomingErrors[t] = sumErr
			}

			// --- PHASE 2: BPTT CALCULATION ---

			dh_next, dC_next := 0.0, 0.0

			// Accumulators
			dW := make([][]float64, 4)
			for i := range dW {
				dW[i] = make([]float64, numInputs+1)
			}
			dB := make([]float64, 4)

			// Buffer for errors to send backward (so we can send them 0->N later)
			// dimensions: [timestep][input_neuron_index]
			dx_buffer := make([][]float64, timesteps)

			n.mu.RLock()
			count := n.HistoryCount
			// deep copy of the slice references (not the LSTMState structs)
			hcopy := make([]*LSTMState, count)
			copy(hcopy, n.History[:count])
			n.mu.RUnlock()

			// Iterate Backwards
			for t := count - 1; t >= 0; t-- {
				st := hcopy[t] //st := n.History[t]
				err_spatial := incomingErrors[t]

				dh_t := err_spatial + dh_next

				tanh_C := tanh(st.C_curr)
				d_o := dh_t * tanh_C * dsigmoid(st.O_gate)
				dC_t := (dh_t * st.O_gate * dtanh(tanh_C)) + dC_next

				d_c_cand := dC_t * st.I_gate * dtanh(st.C_cand)
				d_i := dC_t * st.C_cand * dsigmoid(st.I_gate)
				d_f := dC_t * st.C_prev * dsigmoid(st.F_gate)

				deltas := []float64{d_f, d_i, d_c_cand, d_o}

				for idx, d := range deltas {
					if math.IsNaN(d) || math.IsInf(d, 0) {
						fmt.Printf("CRITICAL: NaN detected at gate %d, timestep %d\n", idx, t)
						fmt.Printf("Inputs: dh_t: %f, dC_t: %f, st.C_curr: %f\n", dh_t, dC_t, st.C_curr)
						// Panic here to stop logs and see the state
						panic("NaN Gradient")
					}
				}

				// Accumulate Gradients
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
					// Snapshot weights before using them
					n.mu.RLock()
					wcopy := make([][]float64, 4)
					for i := range 4 {
						wcopy[i] = slices.Clone(n.Weights[i])
					}
					n.mu.RUnlock()

					// Use snapshot for dx_j
					dx_j := (d_f * wcopy[0][j]) +
						(d_i * wcopy[1][j]) +
						(d_c_cand * wcopy[2][j]) +
						(d_o * wcopy[3][j])

					dx_buffer[t][j] = dx_j
				}

				// Temporal Error
				lastCol := numInputs
				dh_prev_step := (d_f * n.Weights[0][lastCol]) +
					(d_i * n.Weights[1][lastCol]) +
					(d_c_cand * n.Weights[2][lastCol]) +
					(d_o * n.Weights[3][lastCol])

				dh_next = dh_prev_step
				dC_next = dC_t * st.F_gate // Correct: C flows via Forget gate

				// Clean up
				st.X = nil
				statePool.Put(st)
			}

			// --- PHASE 3: ERROR PROPAGATION (CHRONOLOGICAL) ---
			// Send buffered errors in order 0 -> N so the prev layer receives them correctly
			for t := range n.HistoryCount {
				for j := range numInputs {
					n.ErrsToPrev[j] <- dx_buffer[t][j]
				}
			}
			n.mu.Lock()
			// --- PHASE 4: WEIGHT UPDATE ---
			for r := range 4 {
				for c := range numInputs + 1 {
					// 1. Get calculated gradient
					rawGrad := dW[r][c]
					// 2. Clip it
					clippedGrad := clip(rawGrad, n.ClipValue)
					// 3. Apply
					n.Weights[r][c] -= n.LR * clippedGrad
				}
				// Clip Bias as well
				rawBiasGrad := dB[r]
				clippedBiasGrad := clip(rawBiasGrad, n.ClipValue)
				n.Biases[r] -= n.LR * clippedBiasGrad
			}

			// n.History = n.History[:0]
			n.HistoryCount = 0
			n.mu.Unlock()
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
