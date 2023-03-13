package main

// #cgo CFLAGS:   -I. -O3 -DNDEBUG -std=c11 -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3
// #cgo CXXFLAGS: -O3 -DNDEBUG -std=c++11 -fPIC -pthread -I.
// #include "main.h"
import "C"
import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
)

func main() {
	var model string
	var threads, tokens int

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&model, "m", "./models/7B/ggml-model-q4_0.bin", "path to q4_0.bin model file to load")
	flags.IntVar(&threads, "t", 4, "number of threads to use during computation")
	flags.IntVar(&tokens, "n", 128, "number of tokens to predict")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Printf("Parsing program arguments failed: %s", err)
		os.Exit(1)
	}

	state := C.llama_allocate_state()

	fmt.Printf("Loading model %s...\n", model)
	modelPath := C.CString(model)
	success := C.llama_bootstrap(modelPath, state)
	if !success {
		fmt.Println("Loading the model failed")
		os.Exit(1)
	}
	fmt.Printf("Model loaded successfully.\n\n")

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(">>> ")
		text, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				os.Exit(0)
			}
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}

		input := C.CString(text)
		params := C.llama_allocate_params(input, C.int(threads), C.int(tokens))
		result := C.llama_predict(params, state)
		if result == 2 {
			fmt.Println("Predicting failed")
			os.Exit(1)
		}

		C.llama_free_params(params)

		fmt.Printf("\n\n")
	}
}
