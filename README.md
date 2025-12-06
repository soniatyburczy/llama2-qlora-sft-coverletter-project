# llama2-qlora-finetune
End-to-end QLoRA fine-tuning pipeline for LlamA-2-7b-Chat, including dataset cleaning, prompt construction, PEFT configuration, supervised fine-tuning, and evaluation.

## Project Context
This repository contains the code and experiments from my Independent Study at Baruch College (Fall 2025) in collaboration with the Machine Learning & Data Science Club and Math Department.
The goal was to fine-tune a model for structured cover letter generation to be used in a chrome extension.

The code in this repository documents the full experimentation pipeline, including:
- dataset analysis and cleaning
- prompt schema design
- QLoRA/LoRA configuration
- supervised fine-tuning
- evaluation and comparison with the base model

Although this repository is primarily a research record, all code is fully open and can be adapted or reused for your own fine-tuning experiments.
Feel free to explore, modify, or extend the pipeline as needed!

## Model Choice
Llama-2-7b Chat was selected as the base model for various reasons.
A major consideration was balancing memory usage, training stability, and generation quality.
The 7B parameter size sits in an ideal middle ground:
* Large enough to produce coherent, professional writing suitable for cover letters
* Small enough to fine-tune efficiently using QLoRA on limited hardware

Since the goal here is to improve a base model's performance on generating cover letters, an important consideration
is that the model selected should not already be so good at the task that fine-tuning provides no real benefit. 
LLama-2-7b-Chat is a strong general-purpose instruction-following model, 
but it is not specialized for structured, multi-field cover letter generation. Out of the box, it tends to:
* produce generic, template-like responses
* miss important details from the job description or resume
* improvise qualifications that were never provided
* hallucinate experiences or rewrite the applicant's background incorrectly
* ignore or partially use key fields (skills, past experience, required qualifications)

These weaknesses, along with Llama-2-7b's strengths, make it an excellent candidate for fine-tuning.
The chat model was chosen because instruction-tuned models handle structured prompts more reliably, which is important for the task of generating a cover letter
from a resume and job description.

## Pipeline Setup

This project implements a full supervised fine-tuning (SFT) workflow using QLoRA.  
The pipeline consists of the following major components:

### 1. Dataset Loading & Cleaning


### 2. Prompt Schema Design
Each sample is converted into a consistent, structured prompt with the following layout:


## Model Performance Summary

## Example Outputs

## Dataset Challenges & Limitations
Many target examples (~10.2%) in the dataset are low-quality and appear to be AI-generated.
These samples provide weak supervision because they fail to model the structure or grounding required for a personalized cover letter.

**Input Fields (Job + Resume):
```text
Job Title: Machine Learning Engineer
Company: IBM
Preferred Qualifications: PhD in Computer Science

Applicant Name: Oliver Lewis
Past Experience: Software Engineer at Facebook
Current Experience: Machine Learning Engineer at Google
Skills: Python, Tensorflow, Keras, Machine Learning
Qualifications: PhD in Computer Science
```

**Cover Letter:
```text
"I am a skilled Machine Learning Engineer with a PhD in Computer Science. 
have previous experience as a Software Engineer at Facebook and am currently working at Google.
My skillset includes Python, Tensorflow, Keras, and machine learning.
I am excited about this opportunity and believe I would be a great fit for your team."
```
This supervision example is problematic for several reasons:
1. The target text is generic and ungrounded
It does not reference:
* IBM
* job-specific requirements
* role expectations
* the resume fields in any meaningful way
It reads like a template generated without awareness of the input structure.

2. It lacks structural consistency.
A high-quality cover letter should contain:
* a tailored introduction
* justification for fit based on both sides (job + resume)
* role-specific alignment
* a coherent closing

Beyond low-quality target texts, several samples include incorrectly formatted input fields.  
A common problem is the collapse of multi-line or bullet-list job qualifications into a single unstructured line.  
For example, this is an actual entry for the “Preferred Qualifications” field:
```text
experience mentoring other analysts excellent written communication skills excellent sql skills experience with data visualization tools (e.g. looker, tableau, periscope) 5+ years working in an analytics team at a highgrowth company
```

Unfortunately, not many cover letter datasets exist publicly, making this one of the only options despite its various issues.

For future experimentation, further dataset cleaning could improve results, specifically the large increase in variability in quality between generated cover letters.
