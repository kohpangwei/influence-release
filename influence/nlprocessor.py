# from spacy.en import English
# import spacy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import en_core_web_sm

class NLProcessor(object):
    def __init__(self):
        # self.nlp = English()
        self.nlp = en_core_web_sm.load()
        # self.nlp = spacy.load('en_core_web_sm-1.2.0')
        self.vectorizer = CountVectorizer(min_df=5)  
        self.word_vec_len = 300
        
    def process_spam(self, spam, ham):
        """
        Takes in a list of spam emails and a list of ham emails 
        and returns a tuple (docs, Y), where:
        - docs is a list of documents, with each document lemmatized
        and stripped of stop and OOV words.
        - Y is an array of classes {0, 1}. Each element is an example.
        +1 means spam, 0 means ham.
        """
        docs = []
        for raw_doc in spam + ham:
            doc = self.nlp(raw_doc)
            docs.append(' '.join(
                [token.lemma_ for token in doc if (token.is_alpha and not (token.is_oov or token.is_stop))]))

        Y = np.zeros(len(spam) + len(ham))        
        Y[:len(spam)] = 1
        Y[len(spam):] = 0


        docs_Y = zip(docs, Y)
        np.random.shuffle(docs_Y)
        docs, Y = zip(*docs_Y)

        Y = np.array(Y)

        return docs, Y

    def process_newsgroups(self, newsgroups):
        """
        Takes in a newsgroups object returned by fetch_20newsgroups()
        and returns a tuple (docs, Y), where:
        - docs is a list of documents, with each document lemmatized
        and stripped of stop and OOV words.
        - Y is an array of classes {+1, -1}. Each element is an example.
        """
        docs = []
        for raw_doc in newsgroups.data:
            doc = self.nlp(raw_doc)
            docs.append(' '.join(
                [token.lemma_ for token in doc if (token.is_alpha and not (token.is_oov or token.is_stop))]))

        # Convert target to {+1, -1}. It is originally {+1, 0}.
        Y = (np.array(newsgroups.target) * 2) - 1    
        
        return (docs, Y)

    def learn_vocab(self, docs):
        """
        Learns a vocabulary from docs.
        """    
        self.vectorizer.fit(docs)
        
    def get_bag_of_words(self, docs):
        """
        Takes in a list of documents and converts it into a bag of words
        representation. Returns X, a sparse matrix where each row is an example
        and each column is a feature (word in the vocab).
        """
        X = self.vectorizer.transform(docs)
        return X
    
    def get_mean_word_vector(self, docs):
        """
        Takes in a list of documents and returns X, a matrix where each row 
        is an example and each column is the mean word vector in that document.
        """        
        n = len(docs)
        X = np.empty([n, self.word_vec_len])
        doc_vec = np.zeros(self.word_vec_len)
        for idx, doc in enumerate(docs):
            doc_vec = reduce(lambda x, y: x+y, [token.vector for token in self.nlp(doc)])
            doc_vec /= n 
            X[idx, :] = doc_vec
        return X
        