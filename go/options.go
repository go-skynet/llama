package llama

import "runtime"

type PredictOptions struct {
	Seed, Threads, Tokens, TopK, Repeat int
	TopP, Temperature, Penalty          float64
}

type PredictOption func(p *PredictOptions)

var DefaultOptions PredictOptions = PredictOptions{
	Seed:        -1,
	Threads:     runtime.NumCPU(),
	Tokens:      128,
	TopK:        40,
	TopP:        0.95,
	Temperature: 0.80,
	Penalty:     1.3,
	Repeat:      64,
}

// SetSeed sets the random seed for sampling text generation.
func SetSeed(seed int) PredictOption {
	return func(p *PredictOptions) {
		p.Seed = seed
	}
}

// SetThreads sets the number of threads to use for text generation.
func SetThreads(threads int) PredictOption {
	return func(p *PredictOptions) {
		p.Threads = threads
	}
}

// SetTokens sets the number of tokens to generate.
func SetTokens(tokens int) PredictOption {
	return func(p *PredictOptions) {
		p.Tokens = tokens
	}
}

// SetTopK sets the value for top-K sampling.
func SetTopK(topk int) PredictOption {
	return func(p *PredictOptions) {
		p.TopK = topk
	}
}

// SetTopP sets the value for nucleus sampling.
func SetTopP(topp float64) PredictOption {
	return func(p *PredictOptions) {
		p.TopP = topp
	}
}

// SetTemperature sets the temperature value for text generation.
func SetTemperature(temp float64) PredictOption {
	return func(p *PredictOptions) {
		p.Temperature = temp
	}
}

// SetPenalty sets the repetition penalty for text generation.
func SetPenalty(penalty float64) PredictOption {
	return func(p *PredictOptions) {
		p.Penalty = penalty
	}
}

// SetRepeat sets the number of times to repeat text generation.
func SetRepeat(repeat int) PredictOption {
	return func(p *PredictOptions) {
		p.Repeat = repeat
	}
}

// Create a new PredictOptions object with the given options.
func NewPredictOptions(opts ...PredictOption) PredictOptions {
	p := DefaultOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}
