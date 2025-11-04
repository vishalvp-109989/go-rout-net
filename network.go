package main

type Network struct {
	InputLayer  *Layer
	Hidden      []*Layer
	OutputLayer *Layer
}

// NewNetwork builds: InputLayer -> Layer(defs[1]) -> Layer(defs[2]) -> ... -> Layer(defs[len-1]) -> OutputLayer
func NewNetwork(defs ...DenseDef) *Network {
	if len(defs) == 1 && defs[0].InputNeurons != 1 {
		panic("Provide at least an input spec and one Dense layer for m!=1")
	}
	// First Dense must carry input spec (m,n) — ensure user used Dense(1, inputDim)
	if !defs[0].HasInputSpec {
		panic("First Dense must be called with two args: Dense(m, inputDim)")
	}

	// Create Input layer
	iLayer := NewInputLayer(defs[0].InputNeurons, defs[0].Neurons)

	prevErrs := iLayer.ErrsFromNext
	prevIns := iLayer.InsToNext

	var hidden []*Layer

	// For every Dense def except the first (input spec),
	// create a regular Layer (including the last Dense def).
	for i := range len(defs) - 1 {
		l := NewLayer(prevErrs, prevIns, defs[i+1].Neurons)
		hidden = append(hidden, l)

		// advance the prevs to this layer's outputs for the next iteration
		prevErrs = l.ErrsFromNext
		prevIns = l.InsToNext
	}

	// Last layer.
	l := NewLayer(prevErrs, prevIns, 1)
	hidden = append(hidden, l)

	return &Network{
		InputLayer:  iLayer,
		Hidden:      hidden,
		OutputLayer: hidden[len(hidden)-1],
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

	// Calculate the error vector: error = pred - target
	finalErrors := make([]float64, numOutputNeurons)
	for i := range numOutputNeurons {
		finalErrors[i] = pred[i] - target[i]
	}

	// Inject the final error into the output neurons to start backpropagation
	for i := range numOutputNeurons {
		oLayer.Neurons[i].ErrsFromNext[0] <- finalErrors[i]
	}
}

func (nw *Network) WaitForBackpropFinish() {
	iLayer := nw.InputLayer

	// m is the number of rows/parallel inputs
	m := len(iLayer.ErrsFromNext)

	// n is the number of columns/features
	if m == 0 || len(iLayer.ErrsFromNext[0]) == 0 {
		return // Should not happen in a valid network
	}
	n := len(iLayer.ErrsFromNext[0])

	// Wait for the backpropagation signal on ALL m*n channels
	for k := range m {
		for j := range n {
			<-iLayer.ErrsFromNext[k][j]
		}
	}
}
