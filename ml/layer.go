package ml

import (
	"fmt"
)

const (
	LayerTypeDense LayerType = iota
	LayerTypeEmbedding
	LayerTypeLSTM
)

type LayerOption func(*LayerDef)
type LayerType int

type Layer struct {
	Neurons         []*Neuron
	LSTMNeurons     []*LSTMNeuron
	EmbeddingNeuron *EmbeddingNeuron // Only one Embedding Block per Layer
	ErrsFromNext    [][]chan float64
	InsToNext       [][]chan float64
}

type LayerDef struct {
	Type         LayerType // Add this field
	InputNeurons int
	Neurons      int
	Activation   ActivationFunc
	Gradient     ActivationFunc
	HasInputSpec bool
	Initializer  InitializerFunc

	// Embedding Specific Fields
	EmbedDim  int
	VocabSize int
	InputLen  int // InputLen to EmbeddingNeuron

	// LSTM Specific Fields
	Timesteps int // For LSTM layers
}

func Embedding(embedDim int, options ...LayerOption) LayerDef {
	d := LayerDef{
		Type:         LayerTypeEmbedding,
		EmbedDim:     embedDim,
		VocabSize:    0,
		Neurons:      0,
		HasInputSpec: true, // Embedding layer acts as the first hidden layer
		Initializer:  Random,
	}

	for _, opt := range options {
		opt(&d)
	}

	if d.EmbedDim > 0 && d.Neurons > 0 {
		if d.Neurons%d.EmbedDim != 0 {
			panic("OutputDim must be divisible by EmbedDim for Embedding layer")
		}
		// InputLen is the number of words in the sequence
		d.InputLen = d.Neurons / d.EmbedDim
		// InputNeurons is the dimension of the input layer (sequence length)
		d.InputNeurons = d.InputLen
	} else {
		panic("Embedding layer requires EmbedDim, VocabSize, and OutputDim options.")
	}

	return d
}

func VocabSize(size int) LayerOption {
	return func(d *LayerDef) {
		d.VocabSize = size
	}
}

func OutputDim(dim int) LayerOption {
	return func(d *LayerDef) {
		d.Neurons = dim // Output size for the layer definition
	}
}

func LSTM(neurons int, options ...LayerOption) LayerDef {
	d := LayerDef{
		Neurons:     neurons,
		Initializer: Random,
		Type:        LayerTypeLSTM,
	}
	// Apply all functional options
	for _, opt := range options {
		opt(&d)
	}
	return d
}

func Timesteps(timesteps int) LayerOption {
	return func(d *LayerDef) {
		d.Timesteps = timesteps
	}
}

func Dense(neurons int, options ...LayerOption) LayerDef {
	d := LayerDef{
		Neurons:      neurons,
		Activation:   linear,   // Default activation
		Gradient:     dfLinear, // Default derivative
		HasInputSpec: false,
		Initializer:  Random,
		Type:         LayerTypeDense,
	}

	// Apply all functional options
	for _, opt := range options {
		opt(&d)
	}

	return d
}

func Activation(name string) LayerOption {
	act, ok := activationMap[name]
	if !ok {
		panic(fmt.Sprintf("Missing activation function for constant: %s", name))
	}

	return func(d *LayerDef) {
		d.Activation = act.fn
		d.Gradient = act.df
	}
}

func Initializer(name string) LayerOption {
	var weightsInitializer InitializerFunc
	switch name {
	case "xavier":
		weightsInitializer = XavierInit
	case "xavier_normal":
		weightsInitializer = XavierNormal
	case "he":
		weightsInitializer = HeNormal
	case "he_uniform":
		weightsInitializer = HeUniform
	default:
		panic(fmt.Sprintf("Missing initializer function for: %s", name))
	}

	return func(d *LayerDef) {
		d.Initializer = weightsInitializer
	}
}

func InputDim(dim int) LayerOption {
	return func(d *LayerDef) {
		d.InputNeurons = dim
		d.HasInputSpec = true
	}
}

// NewEmbedLayer creates a Layer struct that holds the single Embedding block.
func NewEmbedLayer(errsToPrev, outsFromPrev [][]chan float64, def0, def1 LayerDef) *Layer {
	// For Embedding, the errsToPrev and outsFromPrev should only have 1 row (m=1)
	if len(outsFromPrev) != 1 || len(errsToPrev) != 1 {
		panic("Embedding layer expects input from a single InputLayer row (m=1)")
	}

	layer := &Layer{
		ErrsFromNext: make([][]chan float64, 1),
		InsToNext:    make([][]chan float64, 1),
	}

	outputNeurons := def0.Neurons
	insToNext := make([][]chan float64, def1.Neurons)
	for i := range def1.Neurons {
		insToNext[i] = make([]chan float64, outputNeurons)
		for j := range outputNeurons {
			insToNext[i][j] = make(chan float64, ChannelCapacity) // buffered
		}
	}

	errsFromNext := make([][]chan float64, def1.Neurons)
	for i := range def1.Neurons {
		errsFromNext[i] = make([]chan float64, outputNeurons)
		for j := range outputNeurons {
			// Initialize channels to receive the final error signal from the first hidden layer
			errsFromNext[i][j] = make(chan float64, ChannelCapacity)
		}
	}
	// Create the single Embedding block
	embeddingBlock := NewEmbeddingNeuron(
		def0.EmbedDim,
		def0.VocabSize,
		def0.InputLen,
		errsToPrev[0],
		errsFromNext,
		outsFromPrev[0],
		insToNext,
		def0.Initializer,
	)

	layer.EmbeddingNeuron = embeddingBlock
	layer.ErrsFromNext = errsFromNext
	layer.InsToNext = insToNext

	return layer
}

func NewInputLayer(m, n int) *Layer {
	inToNext := make([][]chan float64, m)
	for i := range m {
		inToNext[i] = make([]chan float64, n)
		for j := range n {
			inToNext[i][j] = make(chan float64, ChannelCapacity) // buffered
		}
	}

	errsFromNext := make([][]chan float64, m)
	for i := range m {
		errsFromNext[i] = make([]chan float64, n)
		for j := range n {
			// Initialize channels to receive the final error signal from the first hidden layer
			errsFromNext[i][j] = make(chan float64, ChannelCapacity)
		}
	}
	return &Layer{
		InsToNext:    inToNext,
		ErrsFromNext: errsFromNext,
	}
}

func NewHiddenLayer(errsToPrev, outsFromPrev [][]chan float64, defCurr, defNext LayerDef) *Layer {
	// 1. Common Validation
	if len(outsFromPrev) != defCurr.Neurons || len(errsToPrev) != defCurr.Neurons {
		panic("Mismatch between provided channels and current layer neuron count")
	}

	if defCurr.Type == LayerTypeLSTM && defCurr.Timesteps != ChannelCapacity {
		panic("LSTM layer's Timesteps must match ChannelCapacity")
	}

	numNeurons := defCurr.Neurons
	outputNeurons := defNext.Neurons

	// 2. Common Struct Initialization
	layer := &Layer{
		ErrsFromNext: make([][]chan float64, numNeurons),
		InsToNext:    make([][]chan float64, numNeurons),
	}

	// 3. Pre-allocate specific neuron slices based on type
	if defCurr.Type == LayerTypeLSTM {
		layer.LSTMNeurons = make([]*LSTMNeuron, numNeurons)
	} else {
		layer.Neurons = make([]*Neuron, numNeurons)
	}

	// 4. Single Loop for Creation
	for j := range numNeurons {
		// A. Create Channels (Identical logic)
		errsFromNext := make([]chan float64, outputNeurons)
		insToNext := make([]chan float64, outputNeurons)

		for i := range outputNeurons {
			errsFromNext[i] = make(chan float64, ChannelCapacity)
			insToNext[i] = make(chan float64, ChannelCapacity)
		}

		// B. Instantiate Specific Neuron
		if defCurr.Type == LayerTypeLSTM {
			layer.LSTMNeurons[j] = NewLSTMNeuron(
				errsToPrev[j], outsFromPrev[j],
				errsFromNext, insToNext,
				defCurr.Timesteps, defCurr.Initializer,
			)
		} else {
			layer.Neurons[j] = NewNeuron(
				errsToPrev[j], outsFromPrev[j],
				errsFromNext, insToNext,
				defCurr.Activation, defCurr.Gradient, defCurr.Initializer,
			)
		}

		// C. Assign Channels (Identical logic)
		layer.ErrsFromNext[j] = errsFromNext
		layer.InsToNext[j] = insToNext
	}

	// 5. Common Transpose
	layer.ErrsFromNext = Transpose(layer.ErrsFromNext)
	layer.InsToNext = Transpose(layer.InsToNext)

	return layer
}

func NewOutputLayer(errsToPrev, outsFromPrev [][]chan float64, def LayerDef) *Layer {
	if len(outsFromPrev) != def.Neurons || len(errsToPrev) != def.Neurons {
		panic("Mismatch between provided channels and current layer neuron count")
	}

	numNeurons := len(outsFromPrev)
	outputNeurons := 1
	f := def.Activation
	df := def.Gradient
	weightsInit := def.Initializer

	layer := &Layer{
		Neurons:      make([]*Neuron, numNeurons),
		ErrsFromNext: make([][]chan float64, numNeurons),
		InsToNext:    make([][]chan float64, numNeurons),
	}

	// 1. Initialize each Neuron in the layer
	for j := range numNeurons {
		errsFromNext := make([]chan float64, outputNeurons)
		insToNext := make([]chan float64, outputNeurons)

		for i := range outputNeurons {
			errsFromNext[i] = make(chan float64, ChannelCapacity)
			insToNext[i] = make(chan float64, ChannelCapacity)
		}
		layer.Neurons[j] = NewNeuron(errsToPrev[j], outsFromPrev[j], errsFromNext, insToNext, f, df, weightsInit)
		layer.ErrsFromNext[j] = errsFromNext
		layer.InsToNext[j] = insToNext
	}

	// 2. Transpose to re-group channels by the next layer's neuron index
	layer.ErrsFromNext = Transpose(layer.ErrsFromNext)
	layer.InsToNext = Transpose(layer.InsToNext)

	return layer
}
