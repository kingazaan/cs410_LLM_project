# CS 410 LLM Project
This is the repository of our CS 410 LLM project. This project transforms the transcripts of Fall 2023 CS 410 online lectures into a dataset and preprocesses it before using it to train a LLM model that can take a CS 410 - related question input and generate answers. Ask it a question about our course, and see how it answers!

# Demo
![alt text](1.gif)
![alt text](2.gif)
![alt text](3.gif)

# Video Presentation
[Please find video presentation here](https://www.youtube.com/watch?v=CdfkFgU7p0s)

# CS 410 Lecture ChatBot Tool
This README is the software documentation for this project.

## Dataset
The `data` folder stores the crude data gleaned from the video transcripts of the CS 410 lectures. `all_lectures.csv` aligns each piece of transcript text with the week number, lesson number and lesson title as introduced in the Fall 2023 offering of CS 410 @ UIUC. These data are to be preprocessed by the LLM model. The `sample_data` folder contains some variants of a segment of our data that can be used for demoing or testing.

## Model
The model implementation and testing lies in the `scripts` folder.
- `testing_llm.ipynb` contains the implementation of the LLM GPT model. It includes code to preprocess the dataset (e.g. word tokenization), the setup of our GPT-based (using `nn` model to decode from massive pre-learned embeddings to pruduce answers to user input) LLM model (with multiple variants, including scaled dot-product attention and multi-head attention), and a `Flask` frontend for users. 
- `testing_QA_finetuning.ipynb` attempts to fine-tune a snapshot of a trained model by creating a QA model (unsuccessful).
- `testing_bigram.ipynb` makes a bigram model (instead of the unigram models in `testing_llm.ipynb`) to parse pairs of consecutive words in the cleaned dataset (used as a test engine).
- `model-0*.pkl` are all the trained model files, saved to a pkl, trained from running the GPT model on the all_lectures.csv dataset. The best iteration at the moment is `model-06.pkl`

# Overview of the Function of the code
This Python Flask app uses our GPT model made from scratch, in the `testing_llm.ipynb` file. This file attempts to create the GPT structure, and utilzing lecture data it runs through the GPT model, and trains on parameters we had chosen for minimal loss. This results in the `model-06.pkl` output, which is then used by the flask frontend. The user inputs a prompt into the frontend, and the text generation takes over and produces an output predicting some of the next words/answers to the prompt. This was trained with a relatively low amount of GPU, and was not pretrained, so outputs are not as strong as for example OpenAI's ChatGPT, yet these outputs are still promising and provide good oversights into lecture data for this class.

# Implementation of Software
This app is generally implemented in the following manner:
Step 1: Mount Google Drive

Step 2: Install Required Packages
Installs the necessary Python packages (torch, transformers) using pip.

Step 3: Load Dataset
Loads a dataset from a CSV file into a Pandas DataFrame.

Step 4: Data Cleaning
Various data cleaning steps using NLTK and regular expressions

Step 5: Tokenization and Filtering
Tokenizes and filters the cleaned transcripts.

Step 6: Embeddings and Vocabulary
Creates word embeddings and builds vocabulary

Step 7: Implement loss function, parts of the block (tranformer class) such as Head, MultiHeadAttention, FeedForward, and finally the GPTLanguageModel Module class which actually embeds the input, passes it through the transformer blocks, and finally generates words as an otuput from the prompt. Configures and sets up the GPT-2 language model. More about this is explained in video here: [Please find video presentation here](https://www.youtube.com/watch?v=CdfkFgU7p0s)
Step 8: Training
Defines Trainer and trains the GPT-2 model

Step 9: Model Evaluation
trainer.evaluate()

Step 10: Save Model

Step 11: Token Generation
Generates tokens using the trained model

Step 12: Flask Frontend
Creates a Flask web application for user interaction

## Evaluation
After releasing this LLM model, we would like to glean feedback from actual users to evaluate the effectiveness of the answers generated by our LLM. We trained a CNN model (in the `CNN` folder) to perform sentiment analysis. Since we don't have actual comments from users yet, we used movie reviews to benchmark the CNN model temporarily.

## Future Works
Future works include, training this dataset after breaking it up into finer segments and running it on a pretrained model in order to fine-tune it. This would probably give us better accuacies than what we currently have, simply becasue of the stronger computing power and larger word corpus to choose from.

# How to Use This Software
## Software packages ussd: 
1. torch (PyTorch)
Overview: PyTorch is an open-source machine learning library that provides a flexible and dynamic computational graph, making it suitable for deep learning applications.
Contribution: PyTorch is the backbone for building neural network models. It facilitates operations on tensors, automatic differentiation, and model training.
2. transformers
Overview: The transformers library by Hugging Face is a powerful tool for working with state-of-the-art natural language processing models, including GPT-2 and BERT.
Contribution: This library simplifies the process of loading pre-trained transformer models, fine-tuning them, and generating text. It provides a high-level interface for transformer-based architectures.
3. transformers[torch]
Overview: This is an additional installation to transformers that ensures compatibility with PyTorch, allowing seamless integration with PyTorch-based models.
Contribution: It enables the use of pre-trained transformer models from the transformers library within PyTorch-based workflows, providing a unified environment for deep learning.
4. pandas
Overview: Pandas is a data manipulation library for Python, providing data structures like DataFrames for efficient data handling and analysis.
Contribution: Pandas is used for loading and preprocessing the dataset stored in a CSV file. It simplifies tasks such as cleaning, filtering, and organizing data into a structured format.
5. numpy
Overview: NumPy is a numerical computing library for Python, offering support for large, multi-dimensional arrays and matrices.
Contribution: Numpy is utilized for numerical operations, especially when working with PyTorch tensors. It enhances the efficiency of mathematical computations in the implementation.
6. regex
Overview: The regex library provides advanced regular expression capabilities, extending the functionality beyond what is available in Python's built-in re module.
Contribution: Regex is employed for text cleaning, allowing sophisticated pattern matching and replacement to preprocess the text data effectively.
7. nltk (Natural Language Toolkit)
Overview: NLTK is a powerful library for natural language processing, offering tools for tasks such as tokenization, stemming, and part-of-speech tagging.
Contribution: NLTK is used for text processing tasks, including tokenization, lemmatization, and stopword removal. It contributes to the cleanliness and structure of the textual data.
8. Flask
Overview: Flask is a lightweight and extensible web framework for Python, suitable for building web applications and APIs.
Contribution: Flask is used to create a user-friendly web interface for interacting with the GPT-2 model. It handles user input, processes requests, and displays model-generated responses.


# Contribution of Each Team Member

All team members contributed equally, but in different ways.
1. Azaan Barlas - Created the GPT model from scratch and trained it to talk to humnas; self-motivated
2. Chris - Created frontend using Flask and integrated it with our GPT model
2. Justin - scraped transcript data (transcripts from all CS 410 lecture videos on Coursera), compiled and edited presentation video
3. Kaiyao - Created the CNN model for future evaluation of our LLM.
