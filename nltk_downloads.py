import nltk
import os

# Directory where NLTK data is stored
nltk_data_dir = nltk.data.find('corpora')

# List of NLTK resources to download
nltk_resources = [
    'punkt',
    'punkt_tab',
    'averaged_perceptron_tagger_eng',
    'maxent_ne_chunker_tab',
    'words',
    'vader_lexicon'
]

# Download each resource if not already downloaded
for resource in nltk_resources:
    try:
        nltk.data.find(resource)
        print(f"{resource} is already downloaded.")
    except LookupError:
        nltk.download(resource)

print("NLTK resource check completed.")