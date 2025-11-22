package data

import (
	"bufio"
	"fmt"
	. "go_rout_net/ml"
	"log"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	// Default tokens
	PAD = "<pad>"
	UNK = "<unk>"

	// Sampling types
	SamplingGreedy  = "greedy"
	SamplingUniform = "uniform"
	SamplingTopK    = "topk"
)

const (
	// ModeAlwaysSample (Case 1): Use the configured SamplingType for ALL tokens.
	ModeAlwaysSample DecodingMode = "always_sample"
	// ModeSampleFirstThenGreedy (Case 2): Use the configured SamplingType for the 1st token, then Greedy for the rest.
	ModeSampleFirstThenGreedy DecodingMode = "sample_first_then_greedy"
	// ModeIntervalSampling (Case 3): Use the configured SamplingType every 'Interval' tokens, otherwise Greedy.
	ModeIntervalSampling DecodingMode = "interval_sampling"
)

var endTokensInf = map[string]bool{
	".":   false,
	"?":   false,
	"!":   false,
	"eos": true,
}

// DecodingMode defines how frequently the configured SamplingType is used.
type DecodingMode string

// DecodingConfig holds the parameters for the inference decoding strategy.
type DecodingConfig struct {
	SamplingType string  // e.g., "greedy", "uniform", "topk" (Used for non-Greedy steps)
	Temperature  float64 // T > 0.
	TopK         int     // The K value for Top-K sampling.

	// New Fields for Mode Control
	Mode       DecodingMode // Defines when to use the SamplingType
	Interval   int          // Used only if Mode is ModeIntervalSampling (e.g., 10)
	Sequential bool         // Whether to use sequential inference (for LSTM)
}

// multinomialSample performs standard sampling from a probability distribution.
// Each class has a chance of being selected proportional to its probability.
func multinomialSample(probs []float64) int {
	r := rand.Float64()
	cumulativeProb := 0.0
	for i, p := range probs {
		cumulativeProb += p
		if r < cumulativeProb {
			return i
		}
	}
	// Fallback in case of floating point inaccuracies, return the last index.
	return len(probs) - 1
}

// greedySample finds the index of the maximum probability.
func greedySample(probs []float64) int {
	maxProb := -1.0
	maxIdx := 0
	for i, p := range probs {
		if p > maxProb {
			maxProb = p
			maxIdx = i
		}
	}
	return maxIdx
}

// topKSample zeros out probabilities outside the top K and then samples multinomial.
func topKSample(probs []float64, K int) int {
	numClasses := len(probs)
	if K <= 0 || K >= numClasses {
		// If K is invalid, fall back to standard multinomial sampling
		return multinomialSample(probs)
	}

	// 1. Find the K largest probabilities and their indices
	type probIndex struct {
		prob float64
		idx  int
	}

	// Create a list of all probabilities with their indices
	indexedProbs := make([]probIndex, numClasses)
	for i, p := range probs {
		indexedProbs[i] = probIndex{prob: p, idx: i}
	}

	// Sort in descending order by probability
	sort.Slice(indexedProbs, func(i, j int) bool {
		return indexedProbs[i].prob > indexedProbs[j].prob
	})

	// 2. Create the masked probabilities and calculate the new sum
	var topKProbs []float64
	var topKIndices []int
	newSum := 0.0

	// Only include the top K
	for i := range K {
		p := indexedProbs[i].prob
		topKProbs = append(topKProbs, p)
		topKIndices = append(topKIndices, indexedProbs[i].idx)
		newSum += p
	}

	// 3. Re-normalize the top K probabilities
	// If newSum is 0 (shouldn't happen with logit subtraction), fall back to uniform.
	if newSum == 0.0 {
		return multinomialSample(probs)
	}

	normalizedProbs := make([]float64, K)
	for i, p := range topKProbs {
		normalizedProbs[i] = p / newSum
	}

	// 4. Sample from the renormalized distribution
	sampledIndexInTopK := multinomialSample(normalizedProbs)

	// 5. Return the original class index
	return topKIndices[sampledIndexInTopK]
}

// --- Main Inference Function ---

// Inference orchestrates the text generation process using the specified decoding strategy.
func Inference(
	nw *Network,
	cfg TrainingConfig,
	decodingCfg DecodingConfig,
	contextLen int,
	wordToID map[string]int,
	idToWord []string,
) {
	reader := bufio.NewReader(os.Stdin)

	log.Printf("Decoding Mode: %+v\n", decodingCfg)
	log.Printf("Prediction Model Type: %s\n", func() string {
		if decodingCfg.Sequential {
			return "Sequential (LSTM/RNN)"
		}
		return "Standard (Feed-Forward)"
	}())

	for {
		fmt.Print("\nEnter text: ")
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(strings.ToLower(line))
		if line == "" {
			continue
		}

		// 1. Tokenize input
		words := strings.Fields(line)

		// 2. Pad or trim to contextLen
		if len(words) < contextLen {
			// Pad with UNK ID's string representation ("<pad>")
			pads := make([]string, contextLen-len(words))
			for i := range pads {
				pads[i] = PAD
			}
			words = append(pads, words...)
		} else if len(words) > contextLen {
			words = words[len(words)-contextLen:] // keep last N words
		}

		// 3. Convert words → IDs (float64 is often used for matrix math compatibility)
		window := make([]float64, contextLen)
		for i := range contextLen {
			w := words[i]
			id, ok := wordToID[w]
			if !ok {
				id = wordToID[UNK]
			}
			window[i] = float64(id)
		}

		// 4. Generate next tokens
		maxTokens := 500 // Use the higher limit from InferenceSeq for maximum generation length
		for i := range maxTokens {
			var logits []float64

			// a) Calculate logits based on model type (The core difference between the two original functions)
			if decodingCfg.Sequential {
				// SEQUENTIAL MODEL LOGIC (e.g., LSTM/RNN)
				// 1. Reset the hidden state before processing the new sequence
				nw.ResetLSTMState()

				// 2. Pass sequence token by token to update state
				for t := range len(window) {
					x_t := []float64{window[t]}
					// PredictProbs updates internal state and returns prediction for next token
					logits = nw.PredictProbs(x_t, cfg)
				}
				// `logits` now holds the prediction based on the full sequence state
			} else {
				// STANDARD MODEL LOGIC (e.g., Feed-Forward)
				// PredictProbs takes the whole context window at once
				logits = nw.PredictProbs(window, cfg)
			}

			// b) Apply Softmax with Temperature
			probs := Softmax(logits, decodingCfg.Temperature)

			// c) Determine the sampling type for the current step
			currentSamplingType := decodingCfg.SamplingType

			switch decodingCfg.Mode {
			case ModeSampleFirstThenGreedy:
				// Use sampling for the first token, then switch to greedy
				if i > 0 {
					currentSamplingType = SamplingGreedy
				}
			case ModeIntervalSampling:
				// Use sampling every N steps, otherwise greedy
				if decodingCfg.Interval > 0 && i%decodingCfg.Interval != 0 {
					currentSamplingType = SamplingGreedy
				}
				// case ModeAlwaysSample: currentSamplingType is already set to decodingCfg.SamplingType
			}

			var predID int
			switch currentSamplingType {
			case SamplingGreedy:
				predID = greedySample(probs)
			case SamplingTopK:
				predID = topKSample(probs, decodingCfg.TopK)
			case SamplingUniform:
				predID = multinomialSample(probs)
			default:
				predID = greedySample(probs) // Fallback
			}

			// d) map ID → word
			word := idToWord[predID]

			// Stop when end-of-sentence or end-of-sequence token appears
			if endTokensInf[word] {
				break
			}

			// Don't print padding and unknown tokens in the output
			if word != PAD && word != UNK {
				fmt.Print(" ", word)
			}

			// e) Slide window: drop first, append predicted ID
			window = append(window[1:], float64(predID))

			// Small delay for better UX
			time.Sleep(100 * time.Millisecond)
		}

		fmt.Println()
	}
}
