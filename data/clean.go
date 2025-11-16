package data

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)
// -------------------------------------------------------------
// Convert low-frequency words to <unk>
// -------------------------------------------------------------
func convertLowFreqsToUnk(freqThreshold int, originalText string, freq map[string]int) string {

	// 1. Build set of low-frequency words
	lowFreqWords := make(map[string]bool)
	for w, c := range freq {
		if c < freqThreshold {
			lowFreqWords[w] = true
		}
	}

	// Tokenize original text with same tokenizer (which preserves <...> tokens)
	re := regexp.MustCompile(`<[^>\s]+>|[a-z0-9]+|[.!?,]`)
	tokens := re.FindAllString(originalText, -1)

	var out strings.Builder
	for i, t := range tokens {
		// If token is a low-frequency plain word (not bracketed), replace it
		// Note: lowFreqWords map contains plain words (no angle brackets)
		if lowFreqWords[t] {
			out.WriteString("<unk>")
		} else {
			out.WriteString(t)
		}
		if i != len(tokens)-1 {
			out.WriteString(" ")
		}
	}

	return out.String()
}

// -------------------------------------------------------------
// CleanData
// -------------------------------------------------------------
func CleanData(contextLen, freqThreshold int, filepath, outputDir string) (
	vocabSize int,
	wordToID map[string]int,
	idToWord []string,
) {

	// ---------------------------------------------------------
	// 1. Read file
	// ---------------------------------------------------------
	data, err := os.ReadFile(filepath)
	if err != nil {
		panic(err)
	}
	text := strings.ToLower(string(data))

	// ---------------------------------------------------------
	// 2. Clean text (allow < and > so tokens like <unk> survive)
	// ---------------------------------------------------------
	text = strings.ReplaceAll(text, "'", "")

	// include '<' and '>' so <unk> remains intact after cleaning
	re := regexp.MustCompile(`[^a-z0-9.!?,<>-]+`)
	text = re.ReplaceAllString(text, " ")

	text = strings.Join(strings.Fields(text), " ")

	// temporary cleaned text (before unk replacement)
	os.WriteFile(outputDir+"cleaned_data.txt", []byte(text), 0644)

	// ---------------------------------------------------------
	// 3. Tokenize once to compute frequency BEFORE unk replacement
	//    Use tokenizer that will preserve existing <...> tokens if any
	// ---------------------------------------------------------
	reTok := regexp.MustCompile(`<[^>\s]+>|[a-z0-9]+|[.!?,]`)
	rawWords := reTok.FindAllString(text, -1)

	// Build frequency table (counts plain tokens as they appear now)
	freq := make(map[string]int)
	for _, w := range rawWords {
		freq[w]++
	}

	// ---------------------------------------------------------
	// 4. Convert LOW-FREQUENCY words to <unk>
	// ---------------------------------------------------------
	// You can make freqThreshold a parameter if you want
	text = convertLowFreqsToUnk(freqThreshold, text, freq)

	// save after unk replacement
	os.WriteFile(outputDir+"cleaned_with_unk.txt", []byte(text), 0644)

	// Tokenize AGAIN after <unk> conversion (this tokenizer treats <unk> as one token)
	words := reTok.FindAllString(text, -1)

	// ---------------------------------------------------------
	// 5. Build vocab (include <pad> and <unk> explicitly)
	// ---------------------------------------------------------
	wordToID = map[string]int{
		"<pad>": 0,
		"<unk>": 1,
	}
	idToWord = []string{"<pad>", "<unk>"}

	for _, w := range words {
		if _, ok := wordToID[w]; !ok {
			wordToID[w] = len(idToWord)
			idToWord = append(idToWord, w)
		}
	}

	vocabSize = len(idToWord)

	// ---------------------------------------------------------
	// 6. Write mapping
	// ---------------------------------------------------------
	f1, _ := os.Create(outputDir + "mapping.txt")
	defer f1.Close()
	for id, w := range idToWord {
		fmt.Fprintf(f1, "%s - %d\n", w, id)
	}

	// ---------------------------------------------------------
	// 7. Create sliding-window dataset
	// ---------------------------------------------------------
	wordDatasetFile, _ := os.Create(outputDir + "dataset_words.csv")
	idDatasetFile, _ := os.Create(outputDir + "dataset.csv")
	defer wordDatasetFile.Close()
	defer idDatasetFile.Close()

	pad := "<pad>"
	endTokens := map[string]bool{
		".": true,
		"!": false,
		"?": false,
	}

	// ---------------------------------------
	// Sentence buffer
	// ---------------------------------------
	var sentence []string

	flushSentence := func(wordsInSentence []string) {

		for i := 0; i < len(wordsInSentence); i++ {
			target := wordsInSentence[i]

			// ------------------------------------------------
			// IMPORTANT RULE: DO NOT predict <unk> as target
			// ------------------------------------------------
			if target == "<unk>" {
				continue
			}

			// build input window with proper indexing
			window := make([]string, contextLen)
			padCount := 0

			for p := 0; p < contextLen; p++ {
				idx := i - contextLen + p
				if idx < 0 {
					window[p] = pad
					padCount++
				} else if idx >= 0 && idx < len(wordsInSentence) {
					window[p] = wordsInSentence[idx]
				} else {
					// out of range (shouldn't normally happen) -> pad
					window[p] = pad
					padCount++
				}
			}

			// skip fully padded row
			if padCount == contextLen {
				continue
			}

			// final row (inputs + target)
			full := append(window, target)

			// write dataset_words.csv
			fmt.Fprintln(wordDatasetFile, strings.Join(full, ","))

			// write dataset.csv (map words to ids)
			idRow := make([]string, len(full))
			for j, w := range full {
				// fallback: if for some reason word not in vocab, use <unk>
				id, ok := wordToID[w]
				if !ok {
					id = wordToID["<unk>"]
				}
				idRow[j] = fmt.Sprintf("%d", id)
			}
			fmt.Fprintln(idDatasetFile, strings.Join(idRow, ","))
		}
	}

	sentence = []string{}

	for _, w := range words {

		if endTokens[w] {
			// include the "." as a final target and flush
			sentence = append(sentence, w)
			flushSentence(sentence)
			sentence = []string{}
			continue
		}

		// otherwise keep accumulating
		sentence = append(sentence, w)
	}

	// flush last sentence if no ending punctuation
	if len(sentence) > 0 {
		flushSentence(sentence)
	}

	return vocabSize, wordToID, idToWord
}
