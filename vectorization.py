from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from numpy import zeros
from numpy import asarray
import os

class VectorStore:
    
    glove_dict = {}
    
    def get_vectors(self, name):
        if name == 'w2v':
            return self.get_w2v_vectorizer()
        elif name == 'glove':
            return self.get_glove_vectorizer()
    
    # also transforms sentences into numbers
    def get_w2v_vectors(self, sentences):
        
        # initialize the word2vec model
        model = Word2Vec(
            vector_size = 300,
            window = 10,
            min_count = 1,
            workers = 4, 
            sg = 1, #0
            epochs = 10
        )
        
        # building the models vocabulary
        _sentences = sentences.apply(word_tokenize)
        model.build_vocab(_sentences)
        
        # training the feature vectors
        model.train(
            _sentences,
            total_examples = model.corpus_count,
            epochs = model.epochs
        )
        
        # convert the feature vectors in KeyedVectors to numpy array
        # tokenize and convert sentences into numbers
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        sentences = tokenizer.texts_to_sequences(sentences)
    
        vocab = tokenizer.word_index
        vocab_size = len(tokenizer.word_index) + 1
        vectors = zeros((vocab_size, model.vector_size))
     
        for word, idx in vocab.items():
            vectors[idx] = model.wv[word]
    
        return vectors, sentences
    
    def get_glove_dictionary(self):
        
        if len(self.glove_dict) != 0:
            return self.glove_dict
        
        with open('../Word Vectors/glove.840B.300d/glove.840B.300d.txt', 'rb') as glove:
            for line in glove:
                line = line.split()
                word = line[0]
                word = str(word).lower()
                vector = asarray(line[1:], 'float32')
                self.glove_dict[word] = vector
         
        return self.glove_dict
     
    # also transforms sentences into numbers
    def get_glove_vectors(self, sentences):
        
        if len(self.glove_dict) == 0:
            self.get_glove_dictionary()
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        sentences = tokenizer.texts_to_sequences(sentences)
    
        vocab = tokenizer.word_index
        vocab_size = len(tokenizer.word_index) + 1
        vectors = zeros((vocab_size, 300))
     
        for word, idx in vocab.items():
            
            word = "b'"+word+"'"
            
            if word in self.glove_dict:
                vectors[idx] = self.glove_dict[word]
            else:
                vectors[idx] = zeros((1, 300))
    
        return vectors, sentences