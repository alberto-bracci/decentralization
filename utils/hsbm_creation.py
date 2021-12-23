import gzip
import pickle
import os
import pandas as pd

import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


## tokenized abstracts
    
def create_tokenized_texts_dict(all_docs_dict, chosen_text_attribute='title'):
    '''
        Creates a dictionary with all the tokenized texts of each paper, using the given chosen_text_attribute. 
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            chosen_text_attribute: attribute in each paper dict to use to get the text layer (str, default:"title")
        
        Returns:
            tokenized_texts_dict: dictionary with paper_id as key and the list of words in the chosen attribute as value (dict of list of strings)
    '''
    # prune text from punctuation and junk...
    wnl = WordNetLemmatizer()
    pattern = re.compile(r'\B#\w*[A-Za-z]+\w*|\b\w*[A-Za-z]+\w*', re.UNICODE)
    def lemmatize(doc):
        '''
            Takes a string doc and returns the list of words, without punctuation and junk
        '''
        l = [wnl.lemmatize(t) for t in pattern.findall(doc)]
        return [w.lower() for w in l if len(w) > 1]

    tokenized_texts_dict = {}
    for paper in all_docs_dict.values():
        if chosen_text_attribute in paper and pd.notna(paper[chosen_text_attribute]) and paper[chosen_text_attribute] != '':
            tokenized_texts_dict[paper['id']] = lemmatize(paper[chosen_text_attribute])
    
    return tokenized_texts_dict
        
def load_tokenized_texts_dict(all_docs_dict, data_folder, chosen_text_attribute='title', file_name = "tokenized_texts_dict.pkl.gz"):
    '''
        Loads or creates&saves the dataframe citations_df with the list of edges of citations between papers within the given dataset.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            chosen_text_attribute: attribute in each paper dict to use to get the text layer (str, default:"title")
            data_folder: path to directory of the data_folder where to save tokenized_texts_dict (str, valid path)
            file_name: file name where to load/save tokenized_texts_dict in data_folder (str, default: "tokenized_texts_dict.pkl.gz")
        
        Returns:
            tokenized_texts_dict: dictionary with paper_id as key and the list of words in the chosen attribute as value (dict of list of strings)
    '''
    try:
        # load from file
        with gzip.open(os.path.join(data_folder,file_name), 'rb') as fp:
            tokenized_texts_dict = pickle.load(fp)
        print("tokenized_texts_dict loaded from file.")
    except:
        print("Creating tokenized_texts_dict from scratch.")
        tokenized_texts_dict = create_tokenized_texts_dict(all_docs_dict, chosen_text_attribute=chosen_text_attribute)
        # dump it
        with gzip.open(os.path.join(data_folder,file_name), 'wb') as fp:
            pickle.dump(tokenized_texts_dict,fp)
        print("tokenized_texts_dict dumped to file.")
    return tokenized_texts_dict


## citations edgelist (hyperlinks)

def create_citations_edgelist(all_docs_dict, papers_with_texts):
    '''
        Creates the dataframe citations_df with the list of edges of citations between papers within the given dataset.
        ACHTUNG: only for those who have a text (in papers_with_texts)
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            papers_with_texts: set or list of paper_ids that have a non empty text (list of strings)
        
        Returns:
            citations_df: pd.DataFrame with columns 'from','to', and each row represents a directed link
    '''
    papers_with_texts = set(papers_with_texts)
    edge_list = []
    for paper_id in papers_with_texts:
        paper = all_docs_dict[paper_id]
        if 'id' in paper and 'inCitations' in paper:
            citations = paper['inCitations']
            for citation in citations:
                if citation in papers_with_texts:
                    edge_list.append((citation,paper['id']))
    
    edge_list = list(set(edge_list))
    citations_df = pd.DataFrame(edge_list,columns = ['from','to'])
    return citations_df


def load_citations_edgelist(all_docs_dict, papers_with_texts, data_folder, file_name = "citations_edgelist.csv"):
    '''
        Loads or creates&saves the dataframe citations_df with the list of edges of citations between papers within the given dataset.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            papers_with_texts: set or list of paper_ids that have a non empty text (list of strings)
            data_folder: path to directory of the data_folder where to save citations_edgelist (str, valid path)
            file_name: file name where to load/save citations_edgelist in data_folder (str, default: "citations_edgelist.csv")
        
        Returns:
            citations_df: pd.DataFrame with columns 'from','to', and each row represents a directed link
    '''
    try:
        # load from file
        citations_df = pd.read_csv(os.path.join(data_folder,file_name))
        print("citations_edgelist loaded from file.")
    except:
        print("Creating citations_edgelist from scratch.")
        citations_df = create_citations_edgelist(all_docs_dict, papers_with_texts)
        # dump it
        citations_df.to_csv(os.path.join(data_folder,file_name),index=False)
        print("citations_edgelist dumped to file.")
    return citations_df





  
    
## Article category (fields of study)

def create_article_category(all_docs_dict):
    '''
        Creates the dataframe citations_df with the list of edges of citations between papers within the given dataset.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
        
        Returns:
            article_category: dictionary with paper_id as key and thestring representing the fields of study of the paper as value (dict of strings)
    '''
    article_category = {}
    for paper_id, paper in all_docs_dict.items():
        if paper['fieldsOfStudy'] is not None:
            article_category[paper_id] = ','.join(paper['fieldsOfStudy'])
        else:
            article_category[paper_id] = 'None'
    return article_category



def load_article_category(all_docs_dict, data_folder, file_name = "article_category.pkl.gz"):
    '''
        Loads or creates&saves a dictionary with all the fields of study.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            data_folder: path to directory of the data_folder where to save citations_edgelist (str, valid path)
            file_name: file name where to load/save citations_edgelist in data_folder (str, default: "citations_edgelist.csv")
        
        Returns:
            article_category: dictionary with paper_id as key and thestring representing the fields of study of the paper as value (dict of strings)
    '''
    try:
        # load from file
        with gzip.open(os.path.join(data_folder,file_name), 'rb') as fp:
            article_category = pickle.load(fp)
        print("article_category loaded from file.")
    except:
        print("Creating article_category from scratch.")
        article_category = create_article_category(all_docs_dict)
        # dump it
        with gzip.open(os.path.join(data_folder,file_name), 'wb') as fp:
            pickle.dump(article_category,fp)
        print("article_category dumped to file.")
    return article_category
