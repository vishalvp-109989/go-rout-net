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

// DecodingMode defines how frequently the configured SamplingType is used.
type DecodingMode string

// DecodingConfig holds the parameters for the inference decoding strategy.
type DecodingConfig struct {
	SamplingType string  // e.g., "greedy", "uniform", "topk" (Used for non-Greedy steps)
	Temperature  float64 // T > 0.
	TopK         int     // The K value for Top-K sampling.

	// New Fields for Mode Control
	Mode     DecodingMode // Defines when to use the SamplingType
	Interval int          // Used only if Mode is ModeIntervalSampling (e.g., 10)
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

	for {
		fmt.Print("\nEnter text: ")
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(strings.ToLower(line))
		if line == "" {
			continue
		}

		// 1. Tokenize input
		words := strings.Fields(line)

		// 2. Pad or trim to contextLen (using ID for UNK for padding)
		if len(words) < contextLen {
			pads := make([]string, contextLen-len(words))
			for i := range pads {
				pads[i] = PAD
			}
			words = append(pads, words...)
		} else if len(words) > contextLen {
			words = words[len(words)-contextLen:] // keep last N words
		}

		// 3. Convert words → IDs
		window := make([]float64, contextLen)
		for i := range contextLen {
			w := words[i]
			id, ok := wordToID[w]
			if !ok {
				id = wordToID[UNK]
			}
			window[i] = float64(id)
		}

		// fmt.Print("Generated: ", strings.Join(words, " "))
		// 4. Generate next tokens
		for i := range 50 {
			// a) Get logits (raw output from NN)
			logits := nw.PredictProbs(window, cfg)

			// b) Apply Softmax with Temperature
			probs := Softmax(logits, decodingCfg.Temperature)

			// c) Determine the sampling type for the current step based on the configured mode.
			currentSamplingType := decodingCfg.SamplingType

			switch decodingCfg.Mode {
			case ModeSampleFirstThenGreedy:
				// Case 2: Just first sampling → use decodecfg.SamplingType, then greedy
				if i > 0 {
					currentSamplingType = SamplingGreedy
				}
			case ModeIntervalSampling:
				// Case 3: Every N samples → use decodecfg.SamplingType, otherwise greedy
				if decodingCfg.Interval > 0 && i%decodingCfg.Interval != 0 {
					currentSamplingType = SamplingGreedy
				}
			case ModeAlwaysSample:
				// Case 1: For every Generate next token -> use decodecfg.SamplingType
				// currentSamplingType is already set to decodingCfg.SamplingType
			}

			var predID int
			switch currentSamplingType {
			case SamplingGreedy:
				predID = greedySample(probs)
			case SamplingTopK:
				// Temperature is applied in Softmax, then TopK masking is applied
				predID = topKSample(probs, decodingCfg.TopK)
			case SamplingUniform:
				// Standard multinomial sampling
				predID = multinomialSample(probs)
			default:
				// Fallback to Greedy
				predID = greedySample(probs)
			}

			// d) map ID → word
			word := idToWord[predID]
			fmt.Print(" ", word)

			// Stop when punctuation appears
			if (word == "." || word == "?" || word == "!") {
				break
			}

			// e) Slide window: drop first, append predicted ID
			window = append(window[1:], float64(predID))

			// Small delay for better UX
			time.Sleep(100 * time.Millisecond)
		}

		fmt.Println()
	}
}
