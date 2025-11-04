package main

type Layer struct {
	Neurons      []*Neuron
	ErrsFromNext [][]chan float64
	InsToNext    [][]chan float64
	GetOutput    func() float64
}

type DenseDef struct {
	InputNeurons int // only used if specified for the first Dense (the input layer)
	Neurons      int // neurons for this Dense block
	HasInputSpec bool
}

func Dense(args ...int) DenseDef {
	if len(args) == 2 {
		return DenseDef{InputNeurons: args[0], Neurons: args[1], HasInputSpec: true}
	}
	if len(args) == 1 {
		return DenseDef{Neurons: args[0], HasInputSpec: false}
	}
	panic("Dense expects 1 or 2 ints")
}

func NewInputLayer(m, n int) *Layer {
	inToNext := make([][]chan float64, m)
	for i := range m {
		inToNext[i] = make([]chan float64, n)
		for j := range n {
			inToNext[i][j] = make(chan float64, BATCH) // buffered
		}
	}

	errsFromNext := make([][]chan float64, m)
	for i := range m {
		errsFromNext[i] = make([]chan float64, n)
		for j := range n {
			// Initialize channels to receive the final error signal from the first hidden layer
			errsFromNext[i][j] = make(chan float64, BATCH)
		}
	}
	return &Layer{
		InsToNext:    inToNext,
		ErrsFromNext: errsFromNext,
	}
}

func NewLayer(errsToPrev, outsFromPrev [][]chan float64, outputNeurons int) *Layer {
	numNeurons := len(outsFromPrev)

	layer := &Layer{
		Neurons:      make([]*Neuron, numNeurons),
		ErrsFromNext: make([][]chan float64, numNeurons),
		InsToNext:    make([][]chan float64, numNeurons),
	}

	// 1. Initialize each Neuron in the layer
	for j := 0; j < numNeurons; j++ {
		errsFromNext := make([]chan float64, outputNeurons)
		insToNext := make([]chan float64, outputNeurons)

		for i := range outputNeurons {
			errsFromNext[i] = make(chan float64, BATCH)
			insToNext[i] = make(chan float64, BATCH)
		}
		layer.Neurons[j] = NewNeuron(errsToPrev[j], outsFromPrev[j], errsFromNext, insToNext)
		layer.ErrsFromNext[j] = errsFromNext
		layer.InsToNext[j] = insToNext
	}

	// 2. Transpose to re-group channels by the next layer's neuron index
	layer.ErrsFromNext = Transpose(layer.ErrsFromNext)
	layer.InsToNext = Transpose(layer.InsToNext)

	return layer
}