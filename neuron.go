package main

import "math/rand"

type Neuron struct {
	Weights []float64
	Bias    float64
	LR      float64

	ErrsToPrev   []chan float64 // send error backward : write
	ErrsFromNext []chan float64 // receive error from layer in front : read

	OutsFromPrev []chan float64 // activated outputs from prev layer  : read
	InsToNext    []chan float64 // activated outputs to next layer : write
}

// Information stored between forward and backward
type Knowledge struct {
	Input []float64
	T     float64
}

// Linear activation and derivative
func linear(x float64) float64   { return x }
func dfLinear(x float64) float64 { return 1 }

func NewNeuron(errsToPrev, outsFromPrev, errsFromNext, insToNext []chan float64) *Neuron {
	n := &Neuron{
		Weights:      make([]float64, len(outsFromPrev)),
		Bias:         0.0,
		LR:           learning_rate,
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
	}

	for i := range n.Weights {
		n.Weights[i] = rand.Float64()*0.2 - 0.1 // small random init
	}

	// Channel for internal forward/backward knowledge
	ch := make(chan Knowledge, BATCH)

	// Forward activation closure
	Activate := func(input []float64) float64 {
		t := n.Bias
		for i, val := range input {
			t += val * n.Weights[i]
		}

		// Store forward knowledge for backward use
		ch <- Knowledge{Input: input, T: t}
		return linear(t)
	}

	// Launch forward worker goroutine
	go func() {
		for {
			in := make([]float64, len(n.OutsFromPrev))
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
		for {
			// wait for forward pass to store state
			kw := <-ch

			// compute local gradient
			grad := dfLinear(kw.T)

			// wait for all incoming errors from front layer
			errFront := 0.0
			for _, outCh := range n.ErrsFromNext {
				errFront += <-outCh
			}

			// send error backward
			for i := range n.ErrsToPrev {
				n.ErrsToPrev[i] <- grad * n.Weights[i] * errFront
			}

			// update weights
			for i := range n.Weights {
				n.Weights[i] -= n.LR * grad * kw.Input[i] * errFront
			}

			n.Bias -= n.LR * grad * errFront
		}
	}()

	return n
}
