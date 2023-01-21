from re import sub
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plot

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('owm-1.4')
# nltk.downlaod('punkt')

class PreProcessor:
    
    english_stopwords = stopwords.words('english')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    def remove_basic_url(self, sentence):
        return sub(r'http\S+', '', sentence)
    
    def remove_non_alphabet(self, sentence):
        return sub(r'[^a-zA-Z]', ' ', sentence)
    
    def make_lowercase(self, sentence):
        return str(sentence).lower()
    
    def get_tokens(self, sentence):
        return word_tokenize(sentence)
    
    def remove_stopwords(self, words):
        return [word for word in words if word not in self.english_stopwords]
    
    def get_stems(self, words):
        return [self.stemmer.stem(word) for word in words]
    
    def get_lemmas(self, words):
        return [self.lemmatizer.lemmatize(word) for word in words]
    
    def remove_words_less_than_minlen(self, words, minlen = 0):
        return [word for word in words if len(word) > minlen]
    
    def get_string(self, words):
        return ' '.join(words)     
    
    def pad(self, sentences):
        
        x = plot.hist([len(sentence) for sentence in sentences], bins = 100, range = [0, 50])
        plot.show()
        plot.xlabel('\nlength of sentence\n')
        plot.ylabel('\nnumber of sentences\n')
        
        _maxlen = int(input("Maximum length of the sentence: "))
        
        sentences = pad_sequences(sentences, maxlen = _maxlen)
        
        return sentences
    
    def deep_preprocess(self, sentence):
        sentence = self.remove_basic_url(sentence)
        sentence = self.remove_non_alphabet(sentence)
        sentence = self.make_lowercase(sentence)
        sentence = self.get_tokens(sentence)
        sentence = self.remove_stopwords(sentence)
        sentence = self.get_lemmas(sentence)
        sentence = self.remove_words_less_than_minlen(sentence, minlen = 1)
        sentence = self.get_string(sentence)
        return sentence