package llama

import "runtime"

type ModelOptions struct {
	ContextSize int
	F16Memory   bool
	Alpaca      bool
	GPT4all     bool
}

type PredictOptions struct {
	Seed, Threads, Tokens, TopK, Repeat int
	TopP, Temperature, Penalty          float64
	IgnoreEOS                           bool
}

type PredictOption func(p *PredictOptions)
type ModelOption func(p *ModelOptions)

var DefaultModelOptions ModelOptions = ModelOptions{
	ContextSize: 512,
	F16Memory:   false,
	Alpaca:      false,
	GPT4all:     false,
}

var DefaultOptions PredictOptions = PredictOptions{
	Seed:        -1,
	Threads:     runtime.NumCPU(),
	Tokens:      128,
	TopK:        10000,
	TopP:        0.90,
	Temperature: 0.96,
	Penalty:     1,
	Repeat:      64,
}

// SetContext sets the context size.
func SetContext(c int) ModelOption {
	return func(p *ModelOptions) {
		p.ContextSize = c
	}
}

var EnableF16Memory ModelOption = func(p *ModelOptions) {
	p.F16Memory = true
}

var EnableAlpaca ModelOption = func(p *ModelOptions) {
	p.Alpaca = true
}

var EnableGPT4All ModelOption = func(p *ModelOptions) {
	p.GPT4all = true
}

// Create a new PredictOptions object with the given options.
func NewModelOptions(opts ...ModelOption) ModelOptions {
	p := DefaultModelOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

var IgnoreEOS PredictOption = func(p *PredictOptions) {
	p.IgnoreEOS = true
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
