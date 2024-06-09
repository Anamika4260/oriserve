import pandas as pd
import spacy
from textblob import TextBlob

# Load the dataset
file_path = '/mnt/data/Evaluation-dataset (1).csv'
df = pd.read_csv(file_path)

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract subthemes and sentiments
def extract_subthemes_and_sentiments(text):
    doc = nlp(text)
    subthemes = []
    for sent in doc.sents:
        aspects = [token.text for token in sent if token.dep_ in ['nsubj', 'dobj'] and token.pos_ in ['NOUN', 'PROPN']]
        sentiment = TextBlob(sent.text).sentiment.polarity
        sentiment_label = 'positive' if sentiment > 0 else 'negative'
        for aspect in aspects:
            subthemes.append((aspect, sentiment_label))
    return subthemes

# Apply the function to the dataset
df['Subthemes_Sentiments'] = df['review'].apply(extract_subthemes_and_sentiments)

# Save the results to a new CSV file
output_file_path = '/mnt/data/subthemes_sentiments_output.csv'
df.to_csv(output_file_path, index=False)

# Display the first few rows of the dataset with subthemes and sentiments
df.head()
