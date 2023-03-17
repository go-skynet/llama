package llama

// #cgo LDFLAGS: -lllama -lm -lstdc++
// #include <lama.h>
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

type LLama struct {
	state unsafe.Pointer
}

func New(model string, ctxSize int) (*LLama, error) {
	if ctxSize == 0 {
		ctxSize = 512
	}
	state := C.llama_allocate_state()
	modelPath := C.CString(model)
	result := C.llama_bootstrap(modelPath, state, C.int(ctxSize))
	if result != 0 {
		return nil, fmt.Errorf("failed loading model")
	}

	return &LLama{state: state}, nil
}

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	out := make([]byte, po.Tokens)
	params := C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat))
	C.llama_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = strings.TrimPrefix(res, " "+text)

	C.llama_free_params(params)

	return res, nil
}
