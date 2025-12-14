package ml

import (
	"fmt"
	"log"
	"math"
)

const (
	LayerTypeInput LayerType = iota // New type for the first definition
	LayerTypeDense
	LayerTypeEmbedding
	LayerTypeLSTM
	LayerTypeConv1D
	LayerTypeConv2D
)

// Helper for logging layer types
func LayerTypeString(t LayerType) string {
	switch t {
	case LayerTypeInput:
		return "Input"
	case LayerTypeDense:
		return "Dense"
	case LayerTypeConv1D:
		return "Conv1D"
	case LayerTypeConv2D:
		return "Conv2D"
	case LayerTypeLSTM:
		return "LSTM"
	case LayerTypeEmbedding:
		return "Embedding"
	default:
		return "Unknown"
	}
}

type LayerType int
type LayerOption func(*LayerDef)

type Layer struct {
	Neurons         []*Neuron
	LSTMNeurons     []*LSTMNeuron
	EmbeddingNeuron *EmbeddingNeuron // Only one Embedding Block per Layer
	ErrsFromNext    [][]chan float64
	InsToNext       [][]chan float64
	LayerDef        LayerDef
}

type LayerDef struct {
	Type         LayerType // Add this field
	Neurons      int
	Activation   ActivationFunc
	Gradient     ActivationFunc
	HasInputSpec bool
	Initializer  InitializerFunc

	// Conv1D Specific Fields
	KernelSize int // K, e.g., 9
	Stride     int // S, e.g., 1

	// Embedding Specific Fields
	EmbedDim  int
	VocabSize int
	InputLen  int // InputLen to EmbeddingNeuron

	// LSTM Specific Fields
	Timesteps int // For LSTM layers
}

func Input(inputDim int, options ...LayerOption) LayerDef {
	d := LayerDef{
		Type:         LayerTypeInput,
		Neurons:      inputDim,
		HasInputSpec: true,
	}

	for _, opt := range options {
		opt(&d)
	}
	return d
}

func Sequential(isSeq bool) LayerOption {
	return func(d *LayerDef) {
		if isSeq {
			if d.Type != LayerTypeInput {
				panic("Sequential option can only be applied to Input layer.")
			}
			d.Neurons = 1 // For sequential input, set neurons to 1
		}
	}
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

func Conv1D(options ...LayerOption) LayerDef {
	d := LayerDef{
		Neurons:     1, // F: number of filters (unsupported option for now)
		KernelSize:  1, // Default kernel size
		Stride:      1, // Default stride
		Initializer: Random,
		Activation:  linear,
		Gradient:    dfLinear,
		Type:        LayerTypeConv1D,
	}
	for _, opt := range options {
		opt(&d)
	}

	if d.KernelSize <= 0 || d.Stride <= 0 {
		panic("Conv1D requires KernelSize and Stride to be positive.")
	}

	return d
}

func Conv2D(options ...LayerOption) LayerDef {
	d := LayerDef{
		Neurons:     1, // This will be recalculated in NewInputLayer based on input size
		KernelSize:  9, // Default to 3x3
		Stride:      1,
		Initializer: Random,
		Activation:  linear,
		Gradient:    dfLinear,
		Type:        LayerTypeConv2D,
	}
	for _, opt := range options {
		opt(&d)
	}

	// Validation 1: Basic Positive Checks
	if d.KernelSize <= 0 || d.Stride <= 0 {
		panic("Conv2D requires KernelSize and Stride to be positive.")
	}

	// Validation 2: Perfect Square Check for Kernel
	// We need to know the side length (e.g., sqrt(9) = 3) to do the stride math later.
	kSide := int(math.Sqrt(float64(d.KernelSize)))
	if kSide*kSide != d.KernelSize {
		panic(fmt.Sprintf("Conv2D Error: KernelSize %d is not a perfect square (e.g., 9 for 3x3, 25 for 5x5).", d.KernelSize))
	}

	return d
}

func Kernel(k int) LayerOption {
	return func(d *LayerDef) { d.KernelSize = k }
}

func Stride(s int) LayerOption {
	return func(d *LayerDef) { d.Stride = s }
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

// NewEmbedLayer creates a Layer struct that holds the single Embedding block.
func NewEmbedLayer(errsToPrev, outsFromPrev [][]chan float64, def0, def1 *LayerDef) *Layer {
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
	log.Printf("Embedding Layer initialized with size %dx%d channels, sized for next layer: %s.\n", 1, outputNeurons, "Flatten")

	// Embedding Neuron does the flatten internally. There is no Layer called "Flatten". Just adding a print here for clarity.
	log.Printf("Flatten Layer initialized with size %dx%d channels, sized for next layer: %s.\n", outputNeurons, def1.Neurons, LayerTypeString(def1.Type))
	return layer
}

func NewInputLayer(defCurr, defNext *LayerDef) *Layer {

	n := defCurr.Neurons
	m := defNext.Neurons

	switch defNext.Type {
	case LayerTypeEmbedding:
		m = 1

	case LayerTypeConv1D:
		inputLen := defCurr.Neurons
		k := defNext.KernelSize
		s := defNext.Stride

		if (inputLen-k)%s != 0 {
			panic(fmt.Sprintf("Conv1D layer error: Input (%d) - Kernel (%d) is not divisible by Stride (%d).", inputLen, k, s))
		}

		m = (inputLen-k)/s + 1
		defNext.Neurons = m
		n = k

	case LayerTypeConv2D:
		inputFlat := defCurr.Neurons // e.g., 784
		kFlat := defNext.KernelSize  // e.g., 9
		s := defNext.Stride          // e.g., 1

		inputSide := int(math.Sqrt(float64(inputFlat)))
		if inputSide*inputSide != inputFlat {
			panic(fmt.Sprintf("Conv2D Error: Input size %d is not a perfect square (cannot infer 2D grid).", inputFlat))
		}

		kSide := int(math.Sqrt(float64(kFlat)))

		if (inputSide-kSide)%s != 0 {
			panic(fmt.Sprintf("Conv2D Error: Input width (%d) - Kernel width (%d) is not divisible by Stride (%d).", inputSide, kSide, s))
		}

		outputSide := (inputSide-kSide)/s + 1 // e.g., (28-3)/1 + 1 = 26

		m = outputSide * outputSide // e.g., 26 * 26 = 676
		defNext.Neurons = m         // Update definition

		n = kFlat // Each neuron looks at 9 inputs (3x3)
	}

	// Create Channel Matrix: [DestinationNeuron][InputIndex]
	inToNext := make([][]chan float64, m)
	errsFromNext := make([][]chan float64, m)

	for i := 0; i < m; i++ {
		inToNext[i] = make([]chan float64, n)
		errsFromNext[i] = make([]chan float64, n)

		for j := 0; j < n; j++ {
			inToNext[i][j] = make(chan float64, ChannelCapacity)
			errsFromNext[i][j] = make(chan float64, ChannelCapacity)
		}
	}

	log.Printf("InputLayer initialized with size %dx%d channels, sized for next layer: %s.\n", n, m, LayerTypeString(defNext.Type))

	return &Layer{
		InsToNext:    inToNext,
		ErrsFromNext: errsFromNext,
		LayerDef:     *defNext,
	}
}

func NewHiddenLayer(errsToPrev, outsFromPrev [][]chan float64, defCurr, defNext *LayerDef) *Layer {
	// 1. Common Validation
	switch defCurr.Type {
	case LayerTypeConv1D, LayerTypeConv2D:
		// Convolutional layers skip the neuron count validation
	case LayerTypeLSTM:
		if defCurr.Timesteps != ChannelCapacity {
			panic("LSTM layer's Timesteps must match ChannelCapacity")
		}
		fallthrough
	default:
		if len(outsFromPrev) != defCurr.Neurons || len(errsToPrev) != defCurr.Neurons {
			panic("Mismatch between provided channels and current layer neuron count")
		}
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
	log.Printf("HiddenLayer (%s) initialized with size %dx%d channels, sized for next layer: %s.\n", LayerTypeString(defCurr.Type), numNeurons, outputNeurons, LayerTypeString(defNext.Type))
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
