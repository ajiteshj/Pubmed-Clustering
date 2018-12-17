from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd


def ret_abstract(pmids):    # Function to retrieve abstracts from pubmed website using Beautiful Soup
    abs_list = []

    for pmid in pmids:
        url = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' + str(pmid)
        response = urlopen(url)         # open URL
        html = response.read()          # read the contents

        parsed_html = BeautifulSoup(html)                 # Parse the html body and obtain abstract contents
        k = parsed_html.body.find('div', attrs={'class': "abstr"})
        if k == None:
            abstract = ''
        else:
            abstract = (k.text[8:])
        abs_list.append(abstract)

    with open('./abstractslist_test.txt', 'a', encoding='utf-8') as fout:   # Write to an out file for future use by clustering program
        for item in abs_list:
            fout.write("%s\n" % item)


    fout.close()


def main():
    f = pd.read_csv('./pmids_test_set_unlabeled.txt', sep= '\t', header= None) # read input file
    ret_abstract(f[0])


if __name__ == '__main__':
    main()

