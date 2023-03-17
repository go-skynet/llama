package llama

// #cgo LDFLAGS: -lllama -lm -lstdc++
// #include <lama.h>
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

var (
	repeatLastN = 64
	seed        = -1
	threads     = 4
	tokens      = 128

	topK          = 40
	topP          = 0.95
	temp          = 0.80
	repeatPenalty = 1.30

	nCtx = 512 // context size

	options = map[string]interface{}{
		"repeat_last_n":  &repeatLastN, // last n tokens to penalize
		"repeat_penalty": &repeatPenalty,
		"seed":           &seed, // RNG seed, -1 will seed based on current time
		"temp":           &temp,
		"threads":        &threads,
		"tokens":         &tokens, // new tokens to predict
		"top_k":          &topK,
		"top_p":          &topP,
	}
)

type LLama struct {
	state unsafe.Pointer
}

func (l *LLama) Load(model string) error {
	state := C.llama_allocate_state()
	modelPath := C.CString(model)
	result := C.llama_bootstrap(modelPath, state, C.int(nCtx))
	if result != 0 {
		return fmt.Errorf("failed loading model")
	}
	l.state = state
	return nil
}

func (l *LLama) Predict(threads int, tokens int, text string) (string, error) {
	input := C.CString(text)
	out := make([]byte, tokens)
	params := C.llama_allocate_params(input, C.int(seed), C.int(threads), C.int(tokens), C.int(topK),
		C.float(topP), C.float(temp), C.float(repeatPenalty), C.int(repeatLastN))
	C.llama_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " "+text)

	C.llama_free_params(params)

	return res, nil
}
