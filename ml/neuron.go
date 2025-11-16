package ml

import (
	"log"
	"math"
	"math/rand"
	"sync"
)

const (
	leak            = 0.01
	channelCapacity = 1
)

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

	InputLen     int // Sequence length (e.g., 5)
	EmbedDim     int // Dimension per word (e.g., 2)
	ConfigUpdate chan NeuronCfg
}

type NeuronCfg struct {
	LR        float64
	BatchSize int
	Ack       chan struct{} // signal completion
}

// Information stored between forward and backward
type Knowledge struct {
	Input []float64
	T     float64
}

func NewNeuron(errsToPrev, outsFromPrev, errsFromNext, insToNext []chan float64, f, df ActivationFunc, init InitializerFunc) *Neuron {
	n := &Neuron{
		Weights:      make([]float64, len(outsFromPrev)),
		Bias:         0.0,
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
		ConfigUpdate: make(chan NeuronCfg), // Buffered channel for new batchSize
	}

	// Init weights
	init(n.Weights, len(n.OutsFromPrev), len(n.InsToNext))

	// Channel for internal forward/backward knowledge
	ch := make(chan *Knowledge, channelCapacity)

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
		kw.Input = input // Set the reference to the input slice
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
		// avgWeightsGrads and avgBiasGrad are moved here and are only allocated once
		// when the goroutine starts, reducing their scope/lifetime issues.
		avgBiasGrad := 0.0
		avgWeightsGrads := make([]float64, len(n.Weights))

		for {
			select {
			case cfg := <-n.ConfigUpdate:
				n.LR = cfg.LR
				n.BatchSize = cfg.BatchSize
				n.BatchCount = 0 // Reset counter
				n.BatchGradsW = make([][]float64, cfg.BatchSize)
				n.BatchGradB = make([]float64, cfg.BatchSize)
				for i := range cfg.BatchSize {
					n.BatchGradsW[i] = make([]float64, len(n.Weights))
				}
				close(cfg.Ack)

			// wait for forward pass to store state
			case kw := <-ch:
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
					for i := range n.Weights {
						n.Weights[i] -= n.LR * grad * kw.Input[i] * errFront
					}
					n.Bias -= n.LR * grad * errFront
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
			}
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
		ConfigUpdate: make(chan NeuronCfg), // Buffered channel for new batchSize
		BatchSize:    1,                    // Default to SGD
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
	ch := make(chan *Knowledge, channelCapacity)

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
			kw.Input = inputIndices // Pass the reference to the reusable buffer

			select {
			case ch <- kw:
				// Stored successfully
			default:
				// Failed to store knowledge (due to capacity), may skip backprop for this step
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
		// GC Optimization: Allocate reusable error buffer ONCE
		m := len(en.ErrsFromNext)
		n := len(en.ErrsFromNext[0])
		errFrontVec := make([]float64, n) // Reusable buffer for aggregated errors
		
		for {
			select {
			case cfg := <-en.ConfigUpdate:
				// --- CONFIG UPDATE STEP ---
				en.LR = cfg.LR
				en.BatchSize = cfg.BatchSize
				en.BatchCount = 0

				// Re-initialize accumulation structures based on new config
				en.BatchGrads = make([][]float64, en.VocabSize)
				for i := range en.VocabSize {
					en.BatchGrads[i] = make([]float64, en.EmbedDim)
				}
				en.BatchUsage = make([]int, en.VocabSize)

				close(cfg.Ack)
			// wait for forward pass to store state
			case kw := <-ch:
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

						// Update the embedding vector immediately (SGD)
						for j := range en.EmbedDim {
							en.Embeddings[idx][j] -= en.LR * gradSlice[j]
						}
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
			}
		}
	}()

	return en
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
