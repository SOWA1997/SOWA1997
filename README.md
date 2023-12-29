import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text, num_sentences=3):
    # Tokeniser le texte en phrases et en mots
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Supprimer les mots vides (stop words)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Calculer la fréquence des mots
    freq = FreqDist(words)

    # Assigner un score à chaque phrase en fonction de la fréquence des mots
    sentence_scores = {}
    for sentence in sentences:
        for word, score in freq.items():
            if word in sentence.lower():
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = score
                else:
                    sentence_scores[sentence] += score

    # Obtenir les phrases avec les scores les plus élevés
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Reconstituer le résumé à partir des phrases sélectionnées
    summarized_text = TreebankWordDetokenizer().detokenize(summarized_sentences)

    return summarized_text

# Exemple d'utilisation
course_text = """
Le changement climatique est un problème mondial qui résulte en grande partie des émissions de gaz à effet de serre causées par l'activité humaine.
Pour atténuer les effets du changement climatique, il est crucial de réduire les émissions de gaz à effet de serre en transitionnant vers des sources d'énergie renouvelable.
Les énergies renouvelables, telles que l'énergie solaire et éolienne, sont des alternatives durables aux combustibles fossiles.
Il est également important de sensibiliser la population à l'importance de la conservation de l'énergie et de la promotion d'un mode de vie durable.
"""

summary = summarize_text(course_text)
print(summary)
