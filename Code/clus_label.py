from sklearn.externals import joblib
from abs_process4 import create_dict
import pandas as pd


km = joblib.load('./pubmed_cluster.pkl')   # Load the saved Kmeans model for labelling
clusters = km.labels_.tolist()             # Obtain Clusters
f = pd.read_csv('./pmids_gold_set_labeled.txt', sep='\t', header=None)
pmids = list(f[0])


data_frame = create_dict(pmids, clusters)       # Create dataframe of pmids and clusters

# Select labels based on top terms in each cluster obtained in previous program ( Change each time the previous code abs_process4.py is run
label_dict = {0:'bardet-biedl syndrome', 1:'triple negative breast cancer', 2:'osteogenesis imperfecta', 3:'spinal muscular atrophy', 4:'cystic fibrosis', 5:'neurofibromatosis'}

with open('./result_train.txt', 'a', encoding='utf-8') as fout:  # write the result to an output file.
    for i, row in data_frame.iterrows():
        fout.write(str(row['title']) + '\t' + str(label_dict[row['cluster']]) + '\n')


