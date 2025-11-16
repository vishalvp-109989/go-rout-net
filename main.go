package main

import (
	. "go_rout_net/data"
	. "go_rout_net/ml"
	"log"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	embedDim := 4
	contextLen := 5

	vocabSize, wordToID, idToWord := CleanData(contextLen, 1, "assets/input.txt", "assets/")
	log.Println("✅ Vocabulary size:", vocabSize)

	outputDim := contextLen * embedDim

	X, Y, err := LoadCSV("assets/dataset.csv")
	if err != nil {
		log.Println("Error:", err)
		return
	}
	// MinMaxNormalize(X)

	inputDim := len(X[0])
	numSamples := len(X)

	log.Printf("Loaded dataset: %d samples, %d input features\n", numSamples, inputDim)

	nw := NewNetwork(
		Embedding(embedDim, VocabSize(vocabSize), OutputDim(outputDim)),
		Dense(128, Activation("relu")),
		Dense(64, Activation("relu")),
		Dense(vocabSize),
	)

	// Try to load previous weights if file exists
	if err := nw.LoadWeights("assets/weights.json"); err != nil {
		log.Println("Error loading weights:", err)
	}

	// Setup Ctrl+C handler
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		log.Println("\nSaving weights before exit...")
		if err := nw.SaveWeights("assets/weights.json"); err != nil {
			log.Println("Error saving weights:", err)
		} else {
			log.Println("Weights saved successfully.")
		}
		os.Exit(0)
	}()

	cfg := TrainingConfig{
		Epochs:       500,
		BatchSize:    1,
		LearningRate: 0.01,
		LossFunction: CATEGORICAL_CROSS_ENTROPY,
		KClasses:     vocabSize, // For CATEGORICAL_CROSS_ENTROPY (Softmax Output).
		VerboseEvery: 100,
	}

	// Train
	nw.Train(X, Y, cfg)

	/* T=1 The model is most predictable. */
	/* T<1 Output is conservative, predictable, and repetitive. The model is highly confident in its top choice. Good for factual answers. */
	/* T>1 Output is creative, diverse, and surprising. The model is more likely to pick lower-ranked, less common words, leading to combinations not in the training set. */
	/* T->0 Always picks the single most probable word. Completely deterministic. */
	/* T->∞ Approaches random selection. */
	decodingCfg := DecodingConfig{
		Mode:         ModeAlwaysSample,
		SamplingType: SamplingGreedy,
		Temperature:  1,
		TopK:         5,
		Interval:     5, // Sample on token 0, 10, 20, 30, 40
	}

	// Inference
	Inference(nw, cfg, decodingCfg, contextLen, wordToID, idToWord)

	// Save weights
	if err := nw.SaveWeights("assets/weights.json"); err != nil {
		log.Println("Error saving weights:", err)
	} else {
		log.Println("Weights saved successfully.")
	}
}
