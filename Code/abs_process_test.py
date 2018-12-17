from __future__ import print_function
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib


stopwords = nltk.corpus.stopwords.words('english')  # Load nltk English stopwords
french_stopwords = nltk.corpus.stopwords.words('french')  # Load nltk French stopwords
rem_words = ['sma','exon', 'mrna', 'smn', 'rna', 'smn2', 'bb', 'bbs1', 'type', 'cf', 'cftr', 'oi','nf1', 'bmd','s','aacr', 'ar', 'asl','nf2', 'promotes', 'al', 'vivo', 'year','sma', 'na', 'mri', 'negative', 'needed', 'newly', 'next', 'treatment', 'number', 'nuclear', 'tnbc', 'patient', 'published', 'including', 'feature', 'study', 'report', 'function', 'cause', 'case', 'associated', 'early', 'new']
stopwords.extend(rem_words)
stopwords.extend(french_stopwords)
lemmatizer = WordNetLemmatizer()

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    sw_filtokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # filter out stopwords
    for t in filtered_tokens:
        if t not in stopwords:
            sw_filtokens.append(t)

    #  lemmatize each word
    lems = [lemmatizer.lemmatize(w) for w in sw_filtokens]
    return lems


def cluster_abs(abstracts, num_clusters, pmids):   # Tf-idf Vectorization and Kmeans- Clustering
    tfidf_vectorizer = TfidfVectorizer(max_df= 0.85, max_features= 200000,
                                       min_df=0.08,use_idf=True, tokenizer=tokenize_only, ngram_range= (1,4))

    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)  # fit the vectorizer to abstracts
    terms = tfidf_vectorizer.get_feature_names()   # Extract feature names

    km = KMeans(n_clusters=num_clusters)   # Kmeans model

    km.fit(tfidf_matrix)   # Fit to tf-idf matrix

    joblib.dump(km, './pubmed_cluster_test.pkl')   # Save the model using joblib because centroids change if we re-run.

    clusters = km.labels_.tolist()    # obtain the clusters
    data_frame = create_dict(pmids, clusters)  # Create a pandas data frame with pmids and their correspnding clusters
    print(data_frame)

    print("Top terms per cluster:")  # Print the top terms per cluster and select the label
    print()
    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print("Cluster %d label:" % i),

        for ind in order_centroids[i, :25]:
            print(' %s' % terms[ind]),

        print()  # add whitespace
        print()  # add whitespace

        print("Cluster %d pmids:" % i, end='')
        for title in data_frame.loc[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print()  # add whitespace
        print()  # add whitespace

    print()
    print()


def create_dict(pmids, clusters):  # function to create data frame for pmids and their clusters

    r_papers = {'title': pmids, 'cluster': clusters}
    frame = pd.DataFrame(r_papers, index=[clusters], columns=['title', 'cluster'])
    return frame


def main():
    f = pd.read_csv('./pmids_test_set_unlabeled.txt', sep='\t', header=None) # read input file (test set)
    pmids = list(f[0])

    with open('./abstractslist_test.txt', 'r', encoding='utf-8') as fin:
        abs_list = fin.read().split('\n')

    num_clusters = 5  # select number of clusters
    cluster_abs(abs_list, num_clusters, pmids)


if __name__ == '__main__':
    main()


