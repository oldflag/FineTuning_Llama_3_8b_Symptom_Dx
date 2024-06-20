# Fine-Tuning Llama3-8b-bnb-4bit Model for Medical Symptom Diagnosis

This project demonstrates how to fine-tune the Llama3-8b-bnb-4bit model using a Question and Answer dataset focused on medical symptoms and their diagnoses.   
The project is implemented using Google Colab and utilizes the `unsloth` library for efficient model handling.

## Overview

The goal of this project is to fine-tune the Llama3-8b-bnb-4bit model to generate accurate medical diagnoses based on input symptoms.   
This is achieved by using a dataset of medical Q&A pairs and adapting the model to understand and respond to medical queries effectively.

## Setup and Installation

1. **Clone the repository and navigate to the project directory:**

   ```bash
   git clone https://github.com/oldfalg/FineTuning_Llama_3_8b_Symptom_Dx.git
   cd FineTuning_Llama_3_8b_Symptom_Dx


## Key Components

•	Model Loading:   
Utilizes the FastLanguageModel from the unsloth library to load the pre-trained Llama3-8b-bnb-4bit model with 4-bit quantization for efficient memory usage.   
•	Dataset Preparation:   
Uses the datasets library to load and process a Q&A dataset for fine-tuning.   
•	Fine-Tuning:   
Fine-tunes the model in Colab to generate accurate diagnoses based on input symptoms.   
•	Model Uploading:   
Supports saving the fine-tuned model in different formats (float16, int4, and LoRA adapters) and uploading it to Hugging Face.   

## Inference

After fine-tuning, the model can be used to generate diagnoses based on new symptom inputs.   
The project supports enabling native faster inference and using the fine-tuned model for generation tasks.
