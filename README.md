# Archived

Use: https://github.com/go-skynet/go-llama.cpp

# llama-go

This is [llama.cpp](https://github.com/ggerganov/llama.cpp) port in golang to use as a library.

## Usage

```
git clone https://github.com/go-skynet/llama.git
cd llama
make libllama.a
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples/main.go -m ggml-alpaca-7b-q4.bin -n 10
```

## Model

For a tiny model, you can use https://github.com/antimatter15/alpaca.cpp . For how to use the prompt, check: https://github.com/tatsu-lab/stanford_alpaca

## License

MIT

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- https://github.com/tatsu-lab/stanford_alpaca
- https://github.com/cornelk/llama-go for the initial ideas
- https://github.com/antimatter15/alpaca.cpp for the light model version (this is compatible and tested only with that checkpoint model!)
