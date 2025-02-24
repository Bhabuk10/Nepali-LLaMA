# Nepali-LLaMA models


This repository contains a collection of Llama models fine-tuned for context-based question answering and instruction-following in the Nepali language. 

---

## Table of Contents
- [Overview](#overview)
- [Model List](#model-list)
- [Details of Fine-tuned Models](#details-of-fine-tuned-models)
- [Datasets Used](#datasets-used)
- [Installation and Usage](#installation-and-usage)


---

## Overview
Each model in this collection has been fine-tuned using Nepali-language datasets and optimized for specific applications, such as handling Nepali conversational data or generating task-specific instructions. These models are provided in various quantization levels, including 16-bit, 4-bit, and LoRA, enabling deployment across diverse hardware setups.


## Model List

| Model Name                                         | Quantization | Hugging Face Link                                                                                   |
|----------------------------------------------------|--------------|----------------------------------------------------------------------------------------------------|
| Llama 3.1 8B Nepali Alpaca Merged                  | 16-bit       | [vhab10/Llama-3.1-8B-Nepali-alpaca-merged-16bit](https://huggingface.co/vhab10/Llama-3.1-8B-Nepali-alpaca-merged-16bit) |
| Llama 3.1 8B Nepali Alpaca GGUF                    | 4-bit GGUF   | [vhab10/Llama-3.1-8B-Nepali-alpaca-gguf_q4_k_m](https://huggingface.co/vhab10/Llama-3.1-8B-Nepali-alpaca-gguf_q4_k_m) |
| Llama 3.1 8B Nepali Alpaca GGUF 16-bit             | 16-bit GGUF  | [vhab10/Llama-3.1-8B-Nepali-alpaca-gguf-16bit](https://huggingface.co/vhab10/Llama-3.1-8B-Nepali-alpaca-gguf-16bit) |
| Llama 3.1 8B Nepali Alpaca                         | 4-bit        | [vhab10/Llama-3.1-8B-Nepali-alpaca-4bit](https://huggingface.co/vhab10/Llama-3.1-8B-Nepali-alpaca-4bit) |
| Llama 3.2 3B Instruct Nepali GGUF                  | 4-bit GGUF   | [vhab10/Llama-3.2-3B-Instruct_Nepali_gguf_q4_k_m](https://huggingface.co/vhab10/Llama-3.2-3B-Instruct_Nepali_gguf_q4_k_m) |
| Llama 3.2 3B Instruct Nepali Merged                | 16-bit       | [vhab10/Llama-3.2-3B-Instruct-Nepali-merged-16bit](https://huggingface.co/vhab10/Llama-3.2-3B-Instruct-Nepali-merged-16bit) |
| Llama 3.2 3B Instruct Nepali                       | 4-bit        | [vhab10/Llama-3.2-3B-Instruct_Nepali_4bit](https://huggingface.co/vhab10/Llama-3.2-3B-Instruct_Nepali_4bit) |
| Llama 3.2 3B Instruct Nepali GGUF 16-bit           | 16-bit GGUF  | [vhab10/Llama-3.2-3B-Instruct_Nepali_gguf_16bit](https://huggingface.co/vhab10/Llama-3.2-3B-Instruct_Nepali_gguf_16bit) |


---

## Details of Fine-tuned Models

### 1. Llama 3.1 8B Series
- **Parameters**: 8 Billion
- **Quantization Options**: 16-bit, 4-bit, GGUF

### 2. Llama 3.2 3B Series
- **Parameters**: 3 Billion
- **Quantization Options**: 16-bit, 4-bit, GGUF

---

## Datasets Used

1. **[alpaca-nepali-cleaned](https://huggingface.co/datasets/saillab/alpaca-nepali-cleaned)**::  A translated version of the Alpaca-52K dataset using Google Translate, released under CC BY-NC for academic and research purposes. It contains 52,002 samples, split into 41.6K training and 10.4K test rows.
2. **[Alpaca Nepali SFT](https://huggingface.co/datasets/Saugatkafley/alpaca-nepali-sft)**: A cleaned Nepali version of the Alpaca dataset, containing 52k samples, specifically aimed at enhancing instruction-following capabilities in Nepali.

---

## Installation and Usage

1. **Install Required Packages**: Ensure `transformers` and `torch` are installed.   
   ```bash
   pip install transformers torch

2. **Model Loading Example**:
   ```bash
   from transformers import AutoModelForCausalLM, AutoTokenizer

   # Load model and tokenizer
   model_name = "vhab10/Llama-3.1-8B-Nepali-alpaca-merged-16bit"  # replace with desired model
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)

   # Inference in Nepali
   input_text = "नेपालको राजधानी के हो?"
   inputs = tokenizer(input_text, return_tensors="pt")
   outputs = model.generate(**inputs)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```
   
3. **Inference Example**

```python

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("vhab10/Llama-3.2-3B-Instruct-Nepali-merged-16bit")
model = AutoModelForCausalLM.from_pretrained("vhab10/Llama-3.2-3B-Instruct-Nepali-merged-16bit")


# Define the Alpaca-style prompt format
alpaca_prompt = """Below is an instruction in Nepali language that describes a task. Write a response in Nepali language that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def generate_response(instruction):
    # Format the prompt with the instruction only
    prompt = alpaca_prompt.format(instruction, "")

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the output with beam search
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,  # Limit response length
        temperature=0.7,     # Optimal temperature for balanced output
        num_beams=5,         # Beam search for better exploration
        top_k=50,            # Limit the sampling pool
        top_p=0.95,          # Use nucleus sampling
        no_repeat_ngram_size=2  # Avoid repetitive phrases
    )

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the response section from the generated text
    response_start = response.find("### Response:") + len("### Response:")
    return response[response_start:].strip()

# Example inputs for diverse and complex tasks
input_queries = [
    "एक आधुनिक मोबाइल फोनको कार्यक्षमताहरूको व्याख्या गर्नुहोस्।",
    "नेपालको प्रमुख पर्व र तिनीहरूको सांस्कृतिक महत्व बताउनुहोस्।",
    "हिमालय पर्वतको महत्त्व जलवायु र पर्यावरणका दृष्टिकोणले वर्णन गर्नुहोस्।",
    "कोभिड-१९ महामारीले शिक्षा प्रणालीमा पारेको प्रभावहरूको सूची तयार गर्नुहोस्।"
]

# Generate responses for the input queries
for query in input_queries:
    print(f"Instruction: {query}")
    response = generate_response(query)
    print(f"Response: {response}\n")

```

   

   
### Disclaimer
The LLaMA models provided in this repository are fine-tuned specifically for question-answering and instruction-following in the Nepali language. While they are designed to provide accurate and contextually relevant responses, their performance may vary due to the limitations of the training data and low computational resource. These models are intended for research and educational purposes and may not fully capture the nuances of all Nepali dialects or complex real-world scenarios. Further training with comprehensive, high-quality Nepali datasets would enhance model accuracy and reliability in broader applications.

