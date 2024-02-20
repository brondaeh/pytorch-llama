# pytorch-llama

## Introduction
A LLaMA 2 model is implemented in the PyTorch framework from scratch.

## Requirements
1. Ensure the required libraries are installed on your system.
```bash
pip install -r requirements.txt
```

2. Download model weights with the official [llama](https://github.com/facebookresearch/llama.git) repository or from [LLaMA 2 7B on Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b). Make sure to include the following:
```bash
pytorch-llama
├── llama-2-7b
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── tokenizer.model
```

## References
1. [pytorch-llama](https://github.com/hkproj/pytorch-llama)
2. [llama](https://github.com/facebookresearch/llama.git)
3. [LLaMA 2 7B on Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b)
4. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
5. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
6. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
7. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)