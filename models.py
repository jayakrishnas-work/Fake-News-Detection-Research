from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Conv1D, MaxPool1D, LSTM, Dropout, Bidirectional, Flatten, BatchNormalization, AveragePooling1D
from sklearn import svm
import os

class ModelStore:
    
    def get_model(self, name, vectors):
        if name == 'Dense NN':
            return self.get_fully(vectors)
        elif name == 'Single LSTM NN':
            return self.get_lstm(vectors)
        elif name == 'Single Bi-LSTM NN':
            return self.get_bi_lstm(vectors)
        elif name == 'CONV + Dense NN':
            return self.get_conv(vectors)
        elif name == 'CONV + Single LSTM NN':
            return self.get_conv_lstm(vectors)
        elif name == 'CONV + Single Bi-LSTM NN':
            return self.get_conv_bi_lstm(vectors)
            
    def get_fully(self, vectors):
        
        fully = Sequential([
            Embedding(
                input_dim = vectors.shape[0],
                output_dim = vectors.shape[1],
                input_length = 30,
                trainable = False,
                weights = [vectors]
            ),
            Dense(16, activation = 'elu'),
            Flatten(),
            Dropout(0.2),
            Dense(6, activation = 'softmax')
        ])
        
        fully.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        
        return fully
            
    def get_lstm(self, vectors):
        
        lstm = Sequential([
            Embedding(
                input_dim = vectors.shape[0],
                output_dim = vectors.shape[1],
                input_length = 30,
                trainable = False,
                weights = [vectors]
            ),
            LSTM(10),
            Dropout(0.2),
            Dense(6, activation = 'softmax')
        ])
        
        lstm.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        
        return lstm
    
    def get_bi_lstm(self, vectors):
        
        bi_lstm = Sequential([
            Embedding(
                input_dim = vectors.shape[0],
                output_dim = vectors.shape[1],
                input_length = 30,
                trainable = False,
                weights = [vectors]
            ),
            Bidirectional(LSTM(10)),
            Dropout(0.2),
            Dense(6, activation = 'softmax')
        ])
        
        bi_lstm.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        
        return bi_lstm
    
    def get_conv(self, vectors):
        
        conv = Sequential([
            Embedding(
                input_dim = vectors.shape[0],
                output_dim = vectors.shape[1],
                input_length = 30,
                trainable = False,
                weights = [vectors]
            ),
            Conv1D(
                filters = 128,
                kernel_size = 3,
                padding = 'same',
                activation = 'elu'
            ),
            # BatchNormalization(),
            AveragePooling1D(pool_size = 2),
            Flatten(),
            Dense(10, activation = 'elu'),
            Dropout(0.2),
            Dense(6, activation = 'softmax')
        ])
        
        conv.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        
        return conv
    
    
    def get_conv_lstm(self, vectors):
        
        conv_lstm = Sequential([
            Embedding(
                input_dim = vectors.shape[0],
                output_dim = vectors.shape[1],
                input_length = 30,
                trainable = False,
                weights = [vectors]
            ),
            Conv1D(
                filters = 128,
                kernel_size = 3,
                padding = 'same',
                activation = 'elu'
            ),
            # BatchNormalization(),
            AveragePooling1D(pool_size = 2),
            LSTM(10),
            Dropout(0.2),
            Dense(6, activation = 'softmax')
        ])
        
        conv_lstm.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        
        return conv_lstm
    
    def get_conv_bi_lstm(self, vectors):
            
        conv_bi_lstm = Sequential([
            Embedding(
                input_dim = vectors.shape[0],
                output_dim = vectors.shape[1],
                input_length = 30,
                trainable = False,
                weights = [vectors]
            ),
            Conv1D(
                filters = 128,
                kernel_size = 3,
                padding = 'same',
                activation = 'elu'
            ),
            # BatchNormalization(),
            AveragePooling1D(pool_size = 2),
            Bidirectional(LSTM(10)),
            Dropout(0.2),
            Dense(6, activation = 'softmax')
        ])
        
        conv_bi_lstm.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )
        
        return conv_bi_lstm