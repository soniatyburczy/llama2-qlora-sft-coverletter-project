# llama2-qlora-finetune
Implementation of a task-specific QLoRA supervised fine-tuning pipeline for LLaMA-2-7B-Chat, developed for an independent study on structured cover letter generation.

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

## Qualitative Examples
### 1. Short-Form Cover Letter

### 2. Long-Form Cover Letter
```text
### Job Description
Job Title:  Data Scientist
Company: DataTech Solutions
Preferred Qualifications:   1. Degree in Statistics, Mathematics, Economics, Computer Science, Data Science, or related field.
  2. 2+ years of experience in data or business analytics, preferably in a digital publisher setting.
  3. Working knowledge of SQL and ability to quickly learn internal tools.
  4. Experience with data visualization tools (Tableau, Looker, Power BI, etc.).
  5. Preferred understanding of the online advertising and publishing industry.

### Applicant Resume
Name: John Smith
Current Experience: Data Scientist at DEF Corp
Past Experience: Data Analyst at XYZ Corp, Business Intelligence Intern at ABC Corp
Skills: Hard skills: Data analysis, statistical modeling, SQL, Python, Tableau
Soft skills: Analytical thinking, problem-solving, communication, adaptability
Qualifications: Bachelor's degree in Statistics, 3+ years of experience in data analysis, proficiency in SQL and Python, strong analytical and communication skills.

### Cover Letter
Using the information above, write a professional, personalized cover letter.
```
<br>

**Base Model Output (Llama-2-7b Chat)**
* Generic phrasing (“I am excited to apply…”)
* Minimal use of job-specific details
* Skill mentions are correct but shallow and not tied to responsibilities
```text
Dear Hiring Manager,

I am excited to apply for the Data Scientist position at DataTech Solutions. As a passionate data analyst with 3+ years of experience in the field, I am confident in my ability to contribute to your team's success.

My current role at DEF Corp has provided me with extensive experience in data analysis, statistical modeling, and SQL. I have a strong understanding of data visualization tools such as Tableau and have used Python to develop predictive models. I am also skilled in communication and problem-solving, having worked with cross-functional teams to present insights and drive business decisions.

I am particularly drawn to DataTech Solutions' focus on leveraging data to drive business growth. I believe that my analytical thinking and problem-solving skills, combined with my proficiency in SQL and Python, make me a strong fit for this role. I am excited about the opportunity to work with a team of talented data scientists and contribute to the company's success.

Thank you for considering my application. I look forward to discussing my qualifications further.

Sincerely,
John Smith
```
0.57 Rougue,0.95 bert precision 0.94 bert recall 0.95 bert f1

<br>

**Fine-Tuned Model Output**
* Incorporates concrete resume details (large datasets, dashboards, reports)
* References job-required tools (SQL, Python, Tableau)
* Stronger narrative of applicant/job fit
```text
Dear Hiring Manager,

I am writing to express my interest in the Data Scientist position at DataTech Solutions. With a strong background in data analysis and a passion for leveraging data to drive insights, I believe I would be a valuable asset to your team.

In my current role as a Data Scientist at DEF Corp, I have successfully analyzed and interpreted large datasets to identify trends and opportunities for growth. I have also developed and maintained dashboards and reports to effectively communicate findings to stakeholders. My proficiency in SQL, Python, and Tableau allows me to efficiently analyze and visualize data, while my strong analytical and problem-solving skills enable me to tackle complex challenges with ease.

I am particularly impressed by DataTech Solutions' commitment to diversity and inclusion, as well as its focus on creating a positive work environment. I am confident that my skills and qualifications align with the requirements of the role, and I am excited about the opportunity to contribute to your team's success.

Thank you for considering my application. I look forward to the possibility of discussing how my skills and experience can benefit DataTech Solutions.

Sincerely,
John Smith
```

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
