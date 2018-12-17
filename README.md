# Pubmed-Clustering
Pubmed clustering problem : Clustering PMIDS by retrieving abstracts from pubmed website - NLP, ML

# README:

This repository contains the following files:
# Input:
pmids_gold_set_labeled.txt
pmids_test_set_unlabeled.txt

# Code:
abstractretrieve.py ( For reading input file and retrieving abstracts from PUBMED website)

For training set i.e pmids_gold_set_labeled.txt because of different parameters
abs_process4.py ( To cluster the abstracts and obtain top terms in each cluster)
clus_label.py (To label the clusters and write to output file. Imports functions from abs_process.py and also the kmeans model saved in previous script is loaded here)

For test set i.e tests_set_unlabeled.txt
abs_process_test.py ( To cluster the abstracts and obtain top terms in each cluster)
clus_label_test.py (To label the clusters and write to output file. Imports functions from abs_process_test.py and also the kmeans model saved in previous script is loaded here) 

# Output:
abstractslist.txt (Abstracts scraped from PUBMED website for training set (gold_set_labeled)
abstractslist_test.txt (Abstracts scraped from PUBMED website for test set (test_set_unlabeled)
result_train.txt (output file for gold_labeled set)
result_test2.txt (output file for unlabeled_test set)


# Packages required:
nltk(tokenizer, lemmatizer, stopwords), sklearn(Kmeans, Tf-idf, joblib) , pandas(read input file and create dataframes), re(regex), urllib.request(URL open), bs4(Beautiful Soup)
From these packages import the required modules or functions.

Can be installed using pip command. 

# Instuctions on building/runnning code:
1. Execute abstractretrieve.py script by changing the input and output file paths for train and test sets. Obtain the abstracts in an outfile.
2. Execute abs_process.py for training set cluster analysis (gold_labeled_set) by giving abstracts file (output of abstractretrieve.py) as input and select labels for clusters from the output printed.
3. Change the label dictionary by giving appropriate labels selected from previous script output, in clus_label.py and run it. Obtain output file in required format. 

Repeat the above steps for test set: Run the scripts abstractretrieve.py, abs_process_test.py and clus_label_test.py. Change the number of clusters to 5 in abs_process_test.py and also appropriate input files.

