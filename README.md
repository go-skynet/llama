# llama-go

Inference of [Facebook's LLaMA](https://github.com/facebookresearch/llama) model in Golang with embedded C/C++.

## Description

This project embeds the work of [llama.cpp](https://github.com/ggerganov/llama.cpp) in a Golang binary.
The main goal is to run the model using 4-bit quantization using CPU on Consumer-Grade hardware.

At startup, the model is loaded and a prompt is offered to enter a prompt,
after the results have been printed another prompt can be entered.
The program can be quit using ctrl+c.

This project was tested on Linux but should be able to get to work on macOS as well.

## Requirements

The memory requirements for the models are approximately:

```
7B  -> 4 GB (1 file)
13B -> 8 GB (2 files)
30B -> 16 GB (4 files)
65B -> 32 GB (8 files)
```

## Installation

```bash
# build this repo
git clone https://github.com/cornelk/llama-go
cd llama-go
make
CGO_CFLAGS_ALLOW='-mf.*' go build .

# install Python dependencies
python3 -m pip install torch numpy sentencepiece
```

Obtain the original LLaMA model weights and place them in ./models - 
for example by using the https://github.com/shawwn/llama-dl script to download them.

Use the following steps to convert the LLaMA-7B model to a format that is compatible:

```bash
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# convert the 7B model to ggml FP16 format
python3 convert-pth-to-ggml.py models/7B/ 1

# quantize the model to 4-bits
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin 2
```

For the bigger models, there are a few extra quantization steps. For example, for LLaMA-13B, converting to FP16 format
will create 2 ggml files, instead of one:

```bash
ggml-model-f16.bin
ggml-model-f16.bin.1
```

You need to quantize each of them separately like this:

```bash
./quantize ./models/13B/ggml-model-f16.bin   ./models/13B/ggml-model-q4_0.bin 2
./quantize ./models/13B/ggml-model-f16.bin.1 ./models/13B/ggml-model-q4_0.bin.1 2
```

When running the larger models, make sure you have enough disk space to store all the intermediate files.

## Usage

```bash
./llama-go -m ./models/13B/ggml-model-q4_0.bin -t 4 -n 128

Loading model ./models/13B/ggml-model-q4_0.bin...
Model loaded successfully.

>>> Some good pun names for a pet groomer:

Some good pun names for a pet groomer:
Rub-a-Dub, Scooby Doo
Hair Force One
Duck and Cover, Two Fleas, One Duck
...

>>>

```
