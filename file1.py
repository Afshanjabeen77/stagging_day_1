#NER
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier = pipeline("ner")
classifier("Hi I Am Afshan")
print(classifier("I am exicte to learn gen Ai!"))

#CLASSIFICATION
classifier = pipeline("zero-shot-classification", candidate_labels=["Geographical", "Political", "General"])
classifier("Mt Everest is the tallest mountain the world")

#TEXT GENERATION
text_generator = pipeline("text-generation")
prompt_text = "iam Afshan and iam from eluru"
text_generator(prompt_text, max_length=100, num_return_sequences=2)[0]['generated_text']

#TEXT SUMMARISATION
summarizer = pipeline("summarization")
text_to_summary = "There are 7 wonders of the world"
summary = summarizer(text_to_summary, max_length=100, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
print(summary[0]['summary_text'])



