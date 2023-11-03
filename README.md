# cs410_LLM_project

Google Doc for our initial Project Report: https://docs.google.com/document/d/1_CWP-kE-oRun291XOF3qJaNMBAOMbvhRzMmDiLPzogY

# Breakdown of project steps:

1. Round up dataset by copying all lectures and titles into a csv or something (lecture title, text) something like that. Clean this data next, will be alot of preprocessing and stuff
2. Need to make up a LOT of QA pairs, aka questions and answers from the dataset. example:
  - Question 4: What is the primary idea behind the Rocchio Feedback method?
  - Answer 4: The primary idea behind the Rocchio Feedback method is to move the query vector closer to the centroid of relevant documents and away from the centroid of non-relevant documents. 
This will take a lot of time, I think we all should get 3/12 weeks each assigned and do this for every lecture. Having like 10 per video is idea, I know it's a lot, but literllay can just use chat gpt or something lol
3. Generate topic modelling from our lectures to fine-tune our model (aka make more columns in our dataset for each lecture, aka "main topics", and this column contains 10 most often keywords from that lecture). Hope that makes sense
4.  Start coding - use the gpt-3 model and load it in as a pretrained model. Then use our data as testing data. Will take a while, so should research this and look into it. Goal is to not use OpenAI's API, only use the pretrained model
5. Integrate LLM into chatbot framework. We'll get there when we get there lol.

# Breakdown of work and Deadlines:
\
* Justin - Grabbing all lecture data
  - November 20th to get all it down
* Kaiyo and Azaan - research best way to fine tune model using QA pairs
  - November 11th Find best wya to create QAs or a better method to fine tune
* Kaiyo and Azaan - reseearch how to ue gpt-3 model (We can try and test some data in this model to create LLM)
  - (November) To be determined, timeline

# Files and Folders:
\
* Data folder

* Folder for QA pairs

* Scripts
  - testing.py script for using gpt-4
