# -*- coding: utf-8 -*-
"""
Created on Sun May 08 23:43:37 2016

@author: Vivek Kalyanarangan
"""

import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from stemming.porter2 import stem
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import hstack
import flask
from sklearn import decomposition
from nltk.tag.perceptron import PerceptronTagger
import nltk
import re
import pickle
import os
import datetime
import time
import requests
import httplib2
import scipy
from scipy.sparse import hstack,csr_matrix,coo_matrix
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
import StringIO

from flask import Flask, jsonify, request, Response
from flasgger import Swagger

app = Flask(__name__)
app.config['SWAGGER'] = {
    "swagger_version": "2.0",
    # headers are optional, the following are default
    # "headers": [
    #     ('Access-Control-Allow-Origin', '*'),
    #     ('Access-Control-Allow-Headers', "Authorization, Content-Type"),
    #     ('Access-Control-Expose-Headers', "Authorization"),
    #     ('Access-Control-Allow-Methods', "GET, POST, PUT, DELETE, OPTIONS"),
    #     ('Access-Control-Allow-Credentials', "true"),
    #     ('Access-Control-Max-Age', 60 * 60 * 24 * 20),
    # ],
    # another optional settings
    # "url_prefix": "swaggerdocs",
    # "subdomain": "docs.mysite,com",
    # specs are also optional if not set /spec is registered exposing all views
    "specs": [
        {
            "version": "2.0.0",
            "title": "Clustering API",
            "endpoint": 'v2_spec',
            "route": '/v2/spec',
            "description": "This API will help you bin individual data points into groups in a guided and unguided manner"
            # rule_filter is optional
            # it is a callable to filter the views to extract

            # "rule_filter": lambda rule: rule.endpoint.startswith(
            #    'should_be_v1_only'
            # )
        }
    ]
}
Swagger(app)

tagger = PerceptronTagger()
tagset = None
stop = nltk.corpus.stopwords
wordnet_lemmatizer = WordNetLemmatizer()

grammar = '''REMOVE: {<PRP><VBP>?<VBG><TO>?}
                         {<PRP><MD><VB><TO>}
                         {<VBZ><DT><JJ>}
                         {<MD><DT><NN>}
                         {<NNP><PRP><VBP>}
                         {<MD><PRP>}
                         {<NNP><PRP><VBP>}
                         {<WDT><MD>}
                         {<PRP><VBP><VBG><VB><DT>}
                         {<VBZ><DT><JJ>}
                         {<VBZ><EX><NN><PRP><VBP><TO><VB>}
                         {<DT><VBZ>}
                         {<PRP><VBP><VBG><TO>}
                         {<MD><VB><TO><VB>}
                         {<VBZ><EX><DT>}
                         {<VB><TO>}
                         {<VBZ>}
                         {<DT>}
                         {<EX>}
                         {<PRP><VBP>}
                         {<CD>}
                         {<PRP\$>}
                         {<PRP>}
                         {<TO>}
                         {<IN>}
                         {<VBP>}
                         {<CC>}
              '''

def stem_doc(x):
    red_text = [stem(word.strip()) for word in x.split(" ") if word.strip()!='']
    return ' '.join(red_text)

def lem(x):
    try:
        return wordnet_lemmatizer.lemmatize(x,pos='v')
    except:
        return x
        
def remove_url(x):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x)

def cleanse_text(text):
    if text:
        text = remove_url(text)
        addl_txt = addl_clean_words(text)
        red_text = clean_words(addl_txt)
        
        no_gram = red_text
        try:
            no_gram = remove_grammar(red_text)
        except:
            no_gram = red_text
    
        #clean = ' '.join([i for i in no_gram.split() if i not in stop])
        if no_gram:
            clean = ' '.join([i for i in no_gram.split()])
            red_text = [lem(word) for word in clean.split(" ")]
            red_text = [stem(word) for word in clean.split(" ")]
            return clean_words(' '.join(red_text))
        else:
            return no_gram
    else:
        return text

def cleanse_text_guided(text):
    if text:
        text = remove_url(text)
        addl_txt = addl_clean_words(text)
        red_text = clean_words_guided(addl_txt)
        
        no_gram = red_text
        try:
            no_gram = remove_grammar(red_text)
        except:
            no_gram = red_text
    
        #clean = ' '.join([i for i in no_gram.split() if i not in stop])
        if no_gram:
            clean = ' '.join([i for i in no_gram.split()])
            red_text = [lem(word) for word in clean.split(" ")]
            red_text = [stem(word) for word in clean.split(" ")]
            return clean_words(' '.join(red_text))
        else:
            return no_gram
    else:
        return text

        
def addl_clean_words(words):
    # any additional data pre-processing
    words = words.replace('can\'t','cannot')
    words = words.replace('won\'t','would not')
    words = words.replace('doesn\'t','does not')
    return words
    
def clean_words(words):
    if words:
        words = remove_email(words)
        words = words.replace('\t',' ')
        words = words.replace(',',' ')
        words = words.replace(':',' ')
        words = words.replace(';',' ')
        words = words.replace('=',' ')
        #words = words.replace('\x92','') # apostrophe encoding
        words = words.replace('\x08','\\b') # \b is being treated as backspace
        #words = ''.join([i for i in words if not i.isdigit()])
        words = words.replace('_',' ')
        words = words.replace('(',' ')
        words = words.replace(')',' ')
        words = words.replace('+',' ')
        words = words.replace('-',' ')
        words = words.replace('`',' ')
        words = words.replace('\'',' ')
        words = words.replace('.',' ')
        words = words.replace('#',' ')
        words = words.replace('/',' ')
        words = words.replace('_',' ')
        words = words.replace('"',' ')
        return words.strip()
    return words

def clean_words_guided(words):
    if words:
        words = remove_email(words)
        words = words.replace('\t',' ')
        words = words.replace(',',' ')
        words = words.replace(':',' ')
        words = words.replace(';',' ')
        words = words.replace('=',' ')
        #words = words.replace('\x92','') # apostrophe encoding
        words = words.replace('\x08','\\b') # \b is being treated as backspace
        #words = ''.join([i for i in words if not i.isdigit()])
        words = words.replace('_',' ')
        words = words.replace('(',' ')
        words = words.replace(')',' ')
        words = words.replace('+',' ')
        words = words.replace('-',' ')
        words = words.replace('`',' ')
        words = words.replace('\'',' ')
        words = words.replace('.',' ')
        words = words.replace('#',' ')
        words = words.replace('/',' ')
        words = words.replace('_',' ')
        words = words.replace('"',' ')
        words = words.replace("'",' ')
        return words.strip()
    return words

    
def remove_grammar(review):
    sentences = nltk.sent_tokenize(review)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    result_review = []
    for sentence in sentences:
        if sentences.strip():
            tagged_review = nltk.tag._pos_tag(sentence, tagset, tagger)
            cp = nltk.RegexpParser(grammar)
            result = cp.parse(tagged_review)
            result_review.append(traverseTree(result))
    return ''.join([word for word in result_review])
    
# Remove email
def remove_email(words):
    mod_words = ''
    if words:
        if words.strip():
            for word in words.split(' '):
                if (word.strip().lower()=='email') or (word.strip().lower()=='phn') or (word.strip().lower()=='phone') or (len(word.strip())<=1):
                    continue
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", word.lower()):
                    mod_words = mod_words+' '+word
                #else:   
    else:
        return words
    return mod_words.strip()
    
def traverseTree(tree):
    imp_words = []
    for n in tree:
        if not isinstance(n, nltk.tree.Tree):               
            if isinstance(n, tuple):
                imp_words.append(n[0])
            else:
                continue
    return ' '.join([word for word in imp_words])

def euc_dist(a,b):
    sum_ = scipy.sparse.csr_matrix.sum(a.multiply(b),axis=1)
    return sum_

@app.route('/unguided_cluster', methods=['POST'])
def index():
    """
    This API will help you generate clusters based on keywords present in unstructured text
    Call this api passing the following parameters - 
        Dataset Path - Choosing the file
        Column Name based on which clustering needs to be done
        Number of Clusters
    Sample URL: http://localhost:8180/cluster/clusters.csv?dataset=\\\\W1400368\\c$\\Users\\VK046010\\Documents\\Python%20Scripts\\RevCycle_PatientAcc.csv&ext=csv&col=SR_SUM_TXT&no_of_clusters=100
    ---
    tags:
      - Clustering API
    parameters:
      - name: dataset
        in: formData
        type: file
        required: true
        description: The fully qualified path of the dataset without the extension.
      - name: col
        in: query
        type: string
        required: true
        description: The column name on which the clustering needs to be done
      - name: no_of_clusters
        in: query
        type: integer
        required: true
        description: The number of clusters
    """
    #file_ = request.args.get('upload')
    #print request.files
    data = pd.read_csv(request.files['dataset'])
    #loc = request.args.get('dataset')
    #ext = loc.split('.')[-1]
    #ext='csv'
    #if 'ext' in request.args:
    #    ext = request.args.get('ext')
    
    unstructure = ''
    if 'col' in request.args:
        unstructure = request.args.get('col')
        print unstructure
    no_of_clusters = 10
    if 'no_of_clusters' in request.args:
        no_of_clusters = int(request.args.get('no_of_clusters'))
    #data=pd.DataFrame()
#    if ext=='csv':
#        data = pd.read_csv(loc)
#    elif ext=='xlsx':
#        data = pd.read_excel(loc)
#    elif ext=='xls':
#        data = pd.read_excel(loc)
        
    data = data.fillna('NULL')
    data['clean_sum'] = data[unstructure].apply(lambda x: cleanse_text(x))
    
    vectorizer = CountVectorizer(analyzer='word',stop_words='english',decode_error='ignore',binary=True)
    #vectorizer.fit(data[unstructure])    
    
    counts = vectorizer.fit_transform(data['clean_sum'])
    
    kmeans = KMeans(n_clusters=no_of_clusters,n_jobs=-1)
    
    data['cluster_num'] = kmeans.fit_predict(counts)
    data = data.drop(['clean_sum'],axis=1)
    output = StringIO.StringIO()
    data.to_csv(output,index=False)
    
    clusters = []
    for i in range(np.shape(kmeans.cluster_centers_)[0]):
        data_cluster = pd.concat([pd.Series(vectorizer.get_feature_names()),pd.DataFrame(kmeans.cluster_centers_[i])],axis=1)
        data_cluster.columns = ['keywords','weights']
        data_cluster = data_cluster.sort_values(by=['weights'],ascending=False)
        data_clust = data_cluster.head(n=10)['keywords'].tolist()
        clusters.append(data_clust)
        #print data_cluster.head(n=10)['keywords']
    #data_CLUSTERS.to_csv('output_full.csv',index=False)
    pd.DataFrame(clusters).to_csv('keywords_.csv')
    data.to_csv('Q2.csv',index=False)
    
    resp = Response(output.getvalue(), mimetype="text/csv")
    resp.headers["Accept"] = "text/csv"
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers["Content-Disposition"] = "attachment; filename=clusters.csv"
    return resp

def phrase_in(x,phrase):
    if phrase in x:
        return True
    else:
        return None

@app.route('/guided_cluster', methods=['POST'])
def index_guided():
    """
    This API will help you generate clusters based on keywords provided by you
    Call this api passing the following parameters - 
        Dataset - The data you want to cluster
        Column Name based on which clustering needs to be done
        Comma separated values of the keywords
    ---
    tags:
      - Clustering API
    parameters:
      - name: dataset
        in: formData
        type: file
        required: true
        description: The dataset
      - name: col
        in: query
        type: string
        required: true
        description: The column name based on which the clustering needs to be done
      - name: phrases
        in: formData
        type: file
        required: true
        description: The keywords for clustering in a single column in a csv
      
    """
    #file_ = request.args.get('upload')
    #print request.files
    data = pd.read_csv(request.files['dataset'])
    data_keywords = pd.read_csv(request.files['phrases'],header=None)
    #loc = request.args.get('dataset')
    #ext = loc.split('.')[-1]
    #ext='csv'
    #if 'ext' in request.args:
    #    ext = request.args.get('ext')
    
    unstructure = ''
    if 'col' in request.args:
        unstructure = request.args.get('col')
  
    data = data.fillna('NULL')
    data['clean_sum'] = data[unstructure].apply(lambda x: cleanse_text(x.lower()))
    #data.to_csv('clean_dat.csv',index=False)
    data_keywords = data_keywords.fillna('NULL')
    data_keywords[data_keywords.columns[0]] = data_keywords[data_keywords.columns[0]].apply(lambda x: str(x).lower())
    data_keywords['clean_keys'] = data_keywords[data_keywords.columns[0]].apply(lambda x: cleanse_text_guided(x))
    vocab_keys = data_keywords['clean_keys'].drop_duplicates().tolist()
    
    counts = np.zeros(shape=(np.shape(data)[0],len(vocab_keys)))
    data_counts = pd.DataFrame(counts,columns=vocab_keys)
    for phrase in vocab_keys:
        data_counts[phrase] = data['clean_sum'].apply(lambda x: phrase_in(x,phrase))
    data = data.drop(['clean_sum'],axis=1)
    data_output = pd.concat([data, data_counts], axis=1)
    output = StringIO.StringIO()
    data_output.to_csv(output,index=False)
 
    resp = Response(output.getvalue(), mimetype="text/csv")
    resp.headers["Accept"] = "text/csv"
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers["Content-Disposition"] = "attachment; filename=clusters.csv"
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=8180,use_evalex=False,threaded=True)