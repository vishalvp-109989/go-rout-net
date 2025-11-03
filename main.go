package main

import (
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

const learning_rate = 0.0001
const useCrossEntropy = false
const K_CLASSES = 1
const BATCH = 1
const EPOCH = 100

func main() {
	X, Y, err := LoadCSV("data.csv")
	if err != nil {
		log.Println("Error:", err)
		return
	}

	inputDim := len(X[0])
	numSamples := len(X[1:])

	log.Printf("Loaded dataset: %d samples, %d input features\n", numSamples, inputDim)

	// m := 1
	// n := inputDim

	network := NewNetwork(
		Dense(16, inputDim),
		Dense(8),
		Dense(4),
		Dense(1),
	)

	// Try to load previous weights if file exists
	if err := network.LoadWeights("weights.json"); err != nil {
		log.Println("Error loading weights:", err)
	}

	// Setup Ctrl+C handler
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		log.Println("\nSaving weights before exit...")
		if err := network.SaveWeights("weights.json"); err != nil {
			log.Println("Error saving weights:", err)
		} else {
			log.Println("Weights saved successfully.")
		}
		os.Exit(0)
	}()

	start := time.Now()
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for epoch := range EPOCH {
			totalLoss := 0.0
			correct := 0
			var predScalar, targetScalar float64
			var predVector, targetVector []float64
			for i := range Y[1:] {
				targetScalar = Y[i]

				x := X[i]
				network.FeedForward(x)

				predVector = network.GetOutput()

				// Target must be converted to One-Hot Encoding (OHE) for Loss & Feedback
				// CROSS-ENTROPY LOSS CALCULATION (FOR K > 1)
				if useCrossEntropy {
					predVector = Softmax(predVector)
					targetVector = make([]float64, K_CLASSES)
					if targetScalar >= 0 && targetScalar < float64(K_CLASSES) {
						targetVector[int(targetScalar)] = 1.0
					}

					loss := 0.0
					// L = - Sum[ y_i * log(y_hat_i) ]
					for k := range K_CLASSES {
						y := targetVector[k]
						y_hat := predVector[k]

						// Add small epsilon to prevent log(0)
						epsilon := 1e-15
						y_hat_safe := math.Max(epsilon, y_hat)

						// The loop naturally only includes the term for the correct class
						// because y_i is 1 only for the correct class (OHE).
						loss += -y * math.Log(y_hat_safe)
					}
					totalLoss += loss
				} else {
					// Mean Squared Error (Regression/K=1) for comparison
					predScalar = predVector[0]
					targetVector = []float64{targetScalar}
					totalLoss += 0.5 * math.Pow(predScalar-targetScalar, 2)
				}

				// 2. BACKWARD PASS INJECTION (Error: y_hat - y)
				network.Feedback(predVector, targetVector)

				// 3. SYNCHRONIZATION BARRIER
				network.WaitForBackpropFinish()

				// Accuracy Check (for classification)
				if useCrossEntropy {
					// Find the index of the highest prediction (predicted class)
					predictedClass := 0
					predVector = Softmax(predVector)
					maxProb := predVector[0]
					for k := 1; k < K_CLASSES; k++ {
						if predVector[k] > maxProb {
							maxProb = predVector[k] // Updates maxProb to the new maximum value
							predictedClass = k      // Updates predictedClass to the index (k) of the new maximum
						}
					}
					for k := 1; k < K_CLASSES; k++ {
						if predVector[k] > maxProb {
							maxProb = predVector[k]
							predictedClass = k
						}
					}
					// Compare predicted class index to the actual class index
					if float64(predictedClass) == targetScalar {
						correct++
					}
				} else {
					// For regression, "accuracy" = % of predictions close to target
					predScalar = predVector[0]
					if math.Abs(predScalar-targetScalar) < 1 {
						correct++
					}
				}
			}
			if epoch%10 == 0 {
				avgLoss := totalLoss / float64(len(Y[1:]))
				acc := float64(correct) / float64(len(Y[1:])) * 100.0
				elapsed := time.Since(start).Minutes()
				log.Printf("Epoch %d | Loss: %.6f | Accuracy: %.2f%% | Time: %.2f min\n", epoch, avgLoss, acc, elapsed)
			}
		}
	}()
	wg.Wait()
	// Test prediction
	// Pick random test data from training set
	rand := rand.Intn(len(X))
	test := X[rand]
	network.FeedForward(test)
	predVector := network.GetOutput()
	if useCrossEntropy {
		// Find the index of the highest prediction (predicted class)
		predVector = Softmax(predVector)
		predictedClass := 0
		maxProb := predVector[0]
		for k := 1; k < K_CLASSES; k++ {
			if predVector[k] > maxProb {
				maxProb = predVector[k] // Updates maxProb to the new maximum value
				predictedClass = k      // Updates predictedClass to the index (k) of the new maximum
			}
		}
		for k := 1; k < K_CLASSES; k++ {
			if predVector[k] > maxProb {
				maxProb = predVector[k]
				predictedClass = k
			}
		}
		log.Printf("Test Input: %v | Predicted Class: %v | Actual Output: %.4f\n", test, predictedClass, Y[rand])

	} else {
		// For regression, "accuracy" = % of predictions close to target
		predScalar := predVector[0]
		log.Printf("Test Input: %v | Predicted Output: %.4f | Actual Output: %.4f\n", test, predScalar, Y[rand])
	}

	if err := network.SaveWeights("weights.json"); err != nil {
		log.Println("Error saving weights:", err)
	} else {
		log.Println("Weights saved successfully.")
	}
}
