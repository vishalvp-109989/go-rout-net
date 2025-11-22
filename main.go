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
	freqThreshold := 1 // Replace words with freq < threshold with <unk>
	sequential := true // Whether to use sequential model (LSTM) or non-sequential (n-gram)

	var outputDim, vocabSize int
	var X [][]float64
	var Y any
	var err error
	var wordToID map[string]int
	var idToWord []string

	if sequential {
		outputDim = embedDim
		ChannelCapacity = contextLen // Adjust channel capacity based on context length for LSTM

		vocabSize, wordToID, idToWord = PreprocessSeqData(
			contextLen,
			freqThreshold,
			"assets/input.txt",
			"assets/",
		)
		X, Y, err = LoadCSVSeq("assets/dataset.csv")
	} else {
		outputDim = contextLen * embedDim // Output dimension for Embedding layer for non-sequential model(n-gram)

		vocabSize, wordToID, idToWord = PreprocessNGramData(
			contextLen,
			freqThreshold,
			"assets/input.txt",
			"assets/",
		)
		X, Y, err = LoadCSV("assets/dataset.csv")
	}

	if err != nil {
		log.Println("Error in Preprocess:", err)
		return
	}
	// MinMaxNormalize(X)

	log.Println("✅ Vocabulary size:", vocabSize)

	inputDim := len(X[0])
	numSamples := len(X)

	log.Printf("Loaded dataset: %d samples, %d input features\n", numSamples, inputDim)

	nw := NewNetwork(
		Embedding(embedDim, VocabSize(vocabSize), OutputDim(outputDim)),
		LSTM(64, Timesteps(contextLen), Initializer("xavier")),
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
		Epochs:       250,
		BatchSize:    1,
		LearningRate: 0.01,
		ClipValue:    5.0,
		LossFunction: CATEGORICAL_CROSS_ENTROPY,
		KClasses:     vocabSize, // For CATEGORICAL_CROSS_ENTROPY (Softmax Output).
		VerboseEvery: 10,
		ShuffleData:  false,
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
		Sequential:   sequential,
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
