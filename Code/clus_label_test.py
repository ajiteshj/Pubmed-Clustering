from sklearn.externals import joblib
from abs_process_test import create_dict
import pandas as pd


km = joblib.load('./pubmed_cluster_test.pkl')   # Load the saved Kmeans model for labelling
clusters = km.labels_.tolist()                  # Obtain Clusters
f = pd.read_csv('./pmids_test_set_unlabeled.txt', sep='\t', header=None)
pmids = list(f[0])


data_frame = create_dict(pmids, clusters)   # Create dataframe of pmids and clusters
print(data_frame)

# Select labels based on top terms in each cluster obtained in previous program ( Change each time the previous code abs_process4.py is run
label_dict = {0:'lynch syndrome', 1:'turner syndrome', 2:'noonan syndrome', 3:'lung adenocarcinoma', 4:'marfan syndrome'}

with open('./result_test2.txt', 'a', encoding='utf-8') as fout:
    for i, row in data_frame.iterrows():    # write the result to an output file.
        fout.write(str(row['title']) + '\t' + str(label_dict[row['cluster']]) + '\n')


