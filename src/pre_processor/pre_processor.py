import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import unicodedata
import re
import spacy

class PreProcessor():
    def __init__(self):
        self.MIN_WORD_FREQUENCY, self.MAX_WORD_FREQUENCY = (10, 100)

        self.stop_words = set(stopwords.words('portuguese'))
        try:
            self.nlp = spacy.load('pt_core_news_sm')
        except:
            spacy.download('pt_core_news_sm')
            self.nlp = spacy.load('pt_core_news_sm')

    def _remove_numbers(self,text):
        return re.sub(r'\S*\d\S*', '', text)

    def _remove_punctuations(self,text):
        return re.sub(r'[^\w\s]+', '', text)

    def _shave(self,text):
        texts = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])

    def _remove_stopwords(self,text):
        words = nltk.word_tokenize(text)
        words = [words for words in words]

        withouts_stopwords = ''
        for word in words:
            if word not in self.stop_words:
                withouts_stopwords += word + ' '
        return withouts_stopwords

    def _lemmatize(self,text):
        lemmas = [token.lemma_ for token in self.nlp(text)]
        return ' '.join(lemmas)

    def pipeline(self, text : str):
        text = self._remove_numbers(text)
        text = text.lower()
        text = self._remove_punctuations(text)
        text = self._shave(text)
        text = self._remove_stopwords(text)
        text = self._lemmatize(text)
        return text