package main

import (
	"math"
	"math/rand"
)

const leak = 0.01

var (
	learningRate    = 0.0
	channelCapacity = 1
	batchSize       = 1
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

func NewNeuron(errsToPrev, outsFromPrev, errsFromNext, insToNext []chan float64, f, df ActivationFunc) *Neuron {
	n := &Neuron{
		Weights:      make([]float64, len(outsFromPrev)),
		Bias:         0.0,
		LR:           learningRate,
		ErrsToPrev:   errsToPrev,
		ErrsFromNext: errsFromNext,
		OutsFromPrev: outsFromPrev,
		InsToNext:    insToNext,
	}

	for i := range n.Weights {
		n.Weights[i] = rand.Float64()*0.2 - 0.1 // small random init
	}

	// Channel for internal forward/backward knowledge
	ch := make(chan Knowledge, channelCapacity)

	// Forward activation closure
	Activate := func(input []float64) float64 {
		t := n.Bias
		for i, val := range input {
			t += val * n.Weights[i]
		}

		// Store forward knowledge for backward use
		select {
		case ch <- Knowledge{Input: input, T: t}:
		default:
			// log.Println("Failed to store knowledge")
		}
		return f(t)
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
			grad := df(kw.T)

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
