import argparse
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model import BM25 
import util
from nltk.corpus import stopwords
import pandas as pd


cleaned_description = util.get_and_clean_data()
stem_cache = util.create_stem_cache(cleaned_description)
stop_dict = set(stopwords.words('English'))
my_custom_processor = util.create_custom_preprocessor(stop_dict, stem_cache)

def search_by_ngram(query):
    vectorizer = CountVectorizer(preprocessor=my_custom_processor, ngram_range=(1,2))
    vectorizer.fit(cleaned_description)
    transformed_data = vectorizer.transform(query)
    return  pd.DataFrame(transformed_data.toarray(), columns=vectorizer.get_feature_names_out())

def search_by_tfidf(query):
    tf_idf_vectorizer = TfidfVectorizer(preprocessor=my_custom_processor, use_idf=True)
    tf_idf_vectorizer.fit(cleaned_description)
    transformed_data = tf_idf_vectorizer.transform(query)
    return pd.DataFrame(transformed_data.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())


parser = argparse.ArgumentParser()

parser.add_argument("--keyword", help="Keyword",type=str) 
parser.add_argument("--algorithm", help="Searching algorithm",type=str) 
args = parser.parse_args()
query = args.keyword
algorithm = args.algorithm

if algorithm == 'ngram' :
    tf_idf_vectorizer = CountVectorizer(preprocessor=my_custom_processor, ngram_range=(1,1))
    X = tf_idf_vectorizer.fit_transform(cleaned_description.astype('U'))
    # Fit the TfIdf model
    query_vec = tf_idf_vectorizer.transform(query.split())
    results = cosine_similarity(X,query_vec).reshape((-1,))
    rank = results.argsort()[-10:][::-1]
    try:
        print(cleaned_description.iloc[rank[:5]])
    except:
        print("Something went wrong")
        
elif algorithm == 'bm25' :
    tf_idf_vectorizer = TfidfVectorizer(preprocessor=my_custom_processor, use_idf=True)
    bm25 = BM25(tf_idf_vectorizer)
    bm25.fit(cleaned_description)
    score = bm25.transform(query)
    rank = np.argsort(score)[::-1]
    try:
        print(cleaned_description.iloc[rank[:5]])
    except:
        print("Something went wrong")
elif algorithm == 'tf-idf':
    tf_idf_vectorizer = TfidfVectorizer(preprocessor=my_custom_processor, use_idf=True)
    X = tf_idf_vectorizer.fit_transform(cleaned_description.astype('U'))
    # Fit the TfIdf model
    query_vec = tf_idf_vectorizer.transform(query.split())
    results = cosine_similarity(X,query_vec).reshape((-1,))
    rank = results.argsort()[-10:][::-1]
    try:
        print(cleaned_description.iloc[rank[:5]])
    except:
        print("Something went wrong")

else: 
    print('Wrong algorithm argument')