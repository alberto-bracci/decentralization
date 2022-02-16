import gzip
import pickle
import os
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
import graph_tool.all as gt
import graph_tool


## tokenized abstracts
    
def create_tokenized_texts_dict(all_docs_dict, 
                                chosen_text_attribute='title', 
                               ):
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
        
def load_tokenized_texts_dict(all_docs_dict, 
                              results_folder, 
                              chosen_text_attribute='title', 
                              file_name = "tokenized_texts_dict.pkl.gz", 
                             ):
    '''
        Loads or creates&saves the dataframe citations_df with the list of edges of citations between papers within the given dataset.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            chosen_text_attribute: attribute in each paper dict to use to get the text layer (str, default:"title")
            results_folder: path to directory of the results_folder where to save tokenized_texts_dict (str, valid path)
            file_name: file name where to load/save tokenized_texts_dict in results_folder (str, default: "tokenized_texts_dict.pkl.gz")
        
        Returns:
            tokenized_texts_dict: dictionary with paper_id as key and the list of words in the chosen attribute as value (dict of list of strings)
    '''
    try:
        # load from file
        with gzip.open(os.path.join(results_folder,file_name), 'rb') as fp:
            tokenized_texts_dict = pickle.load(fp)
        print("tokenized_texts_dict loaded from file.")
    except:
        print("Creating tokenized_texts_dict from scratch.")
        tokenized_texts_dict = create_tokenized_texts_dict(all_docs_dict, chosen_text_attribute=chosen_text_attribute)
        # dump it
        with gzip.open(os.path.join(results_folder,file_name), 'wb') as fp:
            pickle.dump(tokenized_texts_dict,fp)
        print("tokenized_texts_dict dumped to file.")
    return tokenized_texts_dict


## citations edgelist (hyperlinks)

def create_citations_edgelist(all_docs_dict, 
                              papers_with_texts, 
                             ):
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


def load_citations_edgelist(all_docs_dict, 
                            papers_with_texts, 
                            results_folder, 
                            file_name = 'citations_edgelist.csv', 
                           ):
    '''
        Loads or creates&saves the dataframe citations_df with the list of edges of citations between papers within the given dataset.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            papers_with_texts: set or list of paper_ids that have a non empty text (list of strings)
            results_folder: path to directory of the results_folder where to save citations_edgelist (str, valid path)
            file_name: file name where to load/save citations_edgelist in results_folder (str, default: "citations_edgelist.csv")
        
        Returns:
            citations_df: pd.DataFrame with columns 'from','to', and each row represents a directed link
    '''
    try:
        # load from file
        citations_df = pd.read_csv(os.path.join(results_folder,file_name))
        print("citations_edgelist loaded from file.")
    except:
        print("Creating citations_edgelist from scratch.")
        citations_df = create_citations_edgelist(all_docs_dict, papers_with_texts)
        # dump it
        citations_df.to_csv(os.path.join(results_folder,file_name),index=False)
        print("citations_edgelist dumped to file.")
    return citations_df



# Filter the network based on number of cits
def filter_papers_with_cits(all_docs_dict, 
                            papers_with_texts, 
                            results_folder, 
                            file_name_citations_df_no_filter = 'citations_edgelist_all.csv', 
                            min_inCitations=0, 
                           ):
    '''
        Loads or creates&saves the dataframe citations_df with the list of edges of citations between papers within the given dataset. 
        It considers only the papers that are cited or get cited once (in total). It also filters only the papers with at least min_inCitations inCitations.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            papers_with_texts: set or list of paper_ids that have a non empty text (list of strings)
            results_folder: path to directory of the results_folder where to save citations_edgelist (str, valid path)
            file_name: file name where to load/save citations_edgelist in results_folder (str, default: "citations_edgelist.csv")
            min_inCitations: minimum number of inCitations to use during filtering (int, default: 0)
        
        Returns:
            citations_df: pd.DataFrame with columns 'from','to', and each row represents a directed link
    '''
    
    # get all citations_df
    citations_df = load_citations_edgelist(all_docs_dict, papers_with_texts, results_folder, file_name = file_name_citations_df_no_filter)
    
    print('Filter the network',flush=True)
    # 1. keep only articles in the citations layer (citing or cited)
    papers_with_cits = set(list(citations_df['from'].values) + list(citations_df['to'].values))
    # 2. filter out articles with less than min_inCitations overall
    id2NoCits = {x: len(all_docs_dict[x]['inCitations']) for x in all_docs_dict.keys()}
    papers_with_cits_filtered = set([x for x in papers_with_cits if id2NoCits[x]>=min_inCitations])
    # get filtered citations_df
    citations_df = citations_df.loc[(citations_df['from'].isin(papers_with_cits_filtered))&(citations_df['to'].isin(papers_with_cits_filtered))]
    ordered_papers_with_cits = list(set(list(citations_df['from'].values) + list(citations_df['to'].values)))

    return citations_df, ordered_papers_with_cits
    
    
def load_filtered_papers_with_cits(all_docs_dict, 
                                   tokenized_texts_dict, 
                                   results_folder, 
                                   file_name = 'citations_edgelist.csv', 
                                   file_name_citations_df_no_filter = 'citations_edgelist_all.csv', 
                                   min_inCitations=0, 
                                  ):
    '''
        Loads or creates&saves the dataframe citations_df with the list of edges of citations between papers within the given dataset, filtering only papers in the citation network. 
        It considers only the papers that are cited or get cited once (in total). It also filters only the papers with at least min_inCitations inCitations.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            papers_with_texts: set or list of paper_ids that have a non empty text (list of strings)
            results_folder: path to directory of the results_folder where to save citations_edgelist (str, valid path)
            file_name: file name where to load/save citations_edgelist in results_folder (str, default: "citations_edgelist.csv")
            file_name_citations_df_no_filter: file name where to load citations_edgelist_all in results_folder (str, default: "citations_edgelist.csv")
            min_inCitations: minimum number of inCitations to use during filtering (int, default: 0)
        
        Returns:
            citations_df: pd.DataFrame with columns 'from','to', and each row represents a directed link
    '''
    try:
        citations_df = pd.read_csv(os.path.join(results_folder,file_name))
        ordered_papers_with_cits = list(set(list(citations_df['from'].values) + list(citations_df['to'].values)))
    except:
        papers_with_texts = set(list(tokenized_texts_dict.keys()))
        citations_df, ordered_papers_with_cits = filter_papers_with_cits(all_docs_dict, papers_with_texts, results_folder, file_name_citations_df_no_filter, min_inCitations=min_inCitations)
        citations_df.to_csv(os.path.join(results_folder,file_name),index=False)
        
    print('original number of articles',len(all_docs_dict),flush=True)
    print('original number of different words', len(set([item for sublist in tokenized_texts_dict.values() for item in sublist])),flush=True)
    print('filtered number of different words', len(set([item for sublist in [tokenized_texts_dict.get(x) for x in ordered_papers_with_cits] for item in sublist])),flush=True)
    print('filtered number of papers', len(ordered_papers_with_cits),flush=True)
    
    return citations_df, ordered_papers_with_cits




  
    
## Article category (fields of study)

def create_article_category(all_docs_dict, 
                           ):
    '''
        Creates the dataframe article_category with the fields of study of each paper in the given dataset.
        
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



def load_article_category(all_docs_dict, 
                          results_folder, 
                          file_name = "article_category.pkl.gz", 
                         ):
    '''
        Loads or creates&saves a dictionary with the categories of each doc.
        
        Args:
            all_docs_dict: dictionary with id of paper as key and a dict with all the info as value (dict of dicts)
            results_folder: path to directory of the results_folder where to save citations_edgelist (str, valid path)
            file_name: file name where to load/save citations_edgelist in results_folder (str, default: "citations_edgelist.csv")
        
        Returns:
            article_category: dictionary with paper_id as key and thestring representing the fields of study of the paper as value (dict of strings)
    '''
    try:
        # load from file
        with gzip.open(os.path.join(results_folder,file_name), 'rb') as fp:
            article_category = pickle.load(fp)
        print("article_category loaded from file.")
    except:
        print("Creating article_category from scratch.")
        article_category = create_article_category(all_docs_dict)
        # dump it
        with gzip.open(os.path.join(results_folder,file_name), 'wb') as fp:
            pickle.dump(article_category,fp)
        print("article_category dumped to file.")
    return article_category



def filter_dataset(
    all_docs_dict,
    tokenized_texts_dict,
    min_inCitations,
    min_word_occurences,
    results_folder,
    filter_label,
):
# TODO: wrappare in funzione che restituisce IDs, texts e edited texts, fino al try except subito dopo. Check se si possono mettere anche cose di prima. 
# (ordered_papers_with_cits, new_filtered_words, tokenized_texts_dict, results_folder, filter_label)


    # Filter the network

    citations_df, ordered_papers_with_cits = load_filtered_papers_with_cits(all_docs_dict, tokenized_texts_dict, results_folder, min_inCitations=min_inCitations)
    print(f'number of document-document links: {len(citations_df)}',flush=True)
    # words appearing in at least two of these articles
    to_count_words_in_doc_df = pd.DataFrame(data = {'paperId': ordered_papers_with_cits, 'word':[tokenized_texts_dict[x] for x in ordered_papers_with_cits]})
    to_count_words_in_doc_df = to_count_words_in_doc_df.explode('word')
    x = to_count_words_in_doc_df.groupby('word').paperId.nunique()
    new_filtered_words = set(x[x>=min_word_occurences].index.values)
    print('Number of new filtered words', len(new_filtered_words),flush=True)


    # Remove stop words in text data
    try:
        with gzip.open(f'{results_folder}IDs_texts_and_edited_text_papers_with_abstract{filter_label}.pkl.gz', 'rb') as fp:
            IDs,texts,edited_text = pickle.load(fp)
    except:
        IDs = [] # list of unique paperIds
        texts = [] # list of tokenized texts
        for paper_id in ordered_papers_with_cits:
            IDs.append(paper_id)
            texts.append(tokenized_texts_dict[paper_id])

        edited_text = []
        stop_words = stopwords.words('english') + stopwords.words('italian') + stopwords.words('german') + stopwords.words('french') + stopwords.words('spanish')

        # Recall texts is list of lists of words in each document.
        for doc in texts:
            temp_doc = []
            for word in doc:
                if word not in stop_words and word in new_filtered_words:
                    temp_doc.append(word)
            edited_text.append(temp_doc)

    #     with gzip.open(f'{results_folder}edited_text_papers_with_abstract{filter_label}.pkl.gz', 'wb') as fp:
    #             pickle.dump(edited_text,fp)

        with gzip.open(f'{results_folder}IDs_texts_and_edited_text_papers_with_abstract{filter_label}.pkl.gz', 'wb') as fp:
                pickle.dump((IDs,texts,edited_text),fp)
        print('Dumped edited texts')

    print(f'number of word-document links: {np.sum([len(set(x)) for x in edited_text])}',flush=True)

    return (ordered_papers_with_cits, new_filtered_words, tokenized_texts_dict, results_folder, filter_label, IDs, texts, edited_text)




def create_hyperlink_g(
    article_category,
    results_folder,
    filter_label
):
    # Create gt object
    if os.path.exists(f'{results_folder}gt_network{filter_label}.gt'):
        hyperlink_g = gt.load_graph(f'{results_folder}gt_network{filter_label}.gt')
    #     num_vertices = hyperlink_g.num_vertices()
    #     num_edges = hyperlink_g.num_edges()
        label = hyperlink_g.vp['label']
    #     name = hyperlink_g.vp['name']
    #     for v in hyperlink_g.vertices():
    #         label[v] = article_category[name[v]]
        # Retrieve true partition of graph
    #     true_partition = list(hyperlink_g.vp.label)    
        # Retrieve ordering of articles
    #     article_names = list(hyperlink_g.vp.name)
        filename = f'{results_folder}citations_edgelist{filter_label}.csv'
        x = pd.read_csv(filename)
        hyperlinks = [(row[0],row[1]) for source, row in x.iterrows()]  
    #     unique_hyperlinks = hyperlinks.copy()
        print('\nLoaded gt object')
    else:
        print('\nCreating gt object...')
        hyperlink_edgelist_filepath = f'{results_folder}citations_edgelist{filter_label}.csv'
        hyperlink_g = gt.load_graph_from_csv(hyperlink_edgelist_filepath,
                                  skip_first=True,
                                  directed=True,
                                  csv_options={'delimiter': ','},)
    #     num_vertices = hyperlink_g.num_vertices()
    #     num_edges = hyperlink_g.num_edges()

        # Create hyperlinks list
        x = pd.read_csv(hyperlink_edgelist_filepath)
        hyperlinks = [(row[0],row[1]) for source, row in x.iterrows()]  

        label = hyperlink_g.vp['label'] = hyperlink_g.new_vp('string')
        name = hyperlink_g.vp['name'] # every vertex has a name already associated to it!

        # We now assign category article to each Wikipedia article
        for v in hyperlink_g.vertices():
            label[v] = article_category[name[v]]

        # Retrieve true partition of graph
    #     true_partition = list(hyperlink_g.vp.label)    
        # Retrieve ordering of articles
    #     article_names = list(hyperlink_g.vp.name)

    #     unique_hyperlinks = hyperlinks.copy()

        hyperlink_g.save(f'{results_folder}gt_network{filter_label}.gt')

    return (hyperlink_g, hyperlinks)








def load_centralities(
    all_docs_dict,
    citations_df,
    ordered_paper_ids,
    hyperlink_g, 
    results_folder,
    filter_label = '',
):
    try:
        with gzip.open(f'{results_folder}hyperlink_g_centralities{filter_label}.pkl.gz','rb') as fp:
            centralities = pickle.load(fp)
        print('\nLoaded centralities.')
    except:
        print('\nCalculating centralities...')
        centralities = compute_centralities(
            all_docs_dict=all_docs_dict,
            citations_df=citations_df,
            ordered_paper_ids=ordered_paper_ids,
            hyperlink_g=hyperlink_g, 
            results_folder=results_folder,
            filter_label=filter_label)
    return centralities

def compute_centralities(
    all_docs_dict,
    citations_df,
    ordered_paper_ids,
    hyperlink_g, 
    results_folder,
    filter_label = '',
):
    id2NoCits = {x: len(all_docs_dict[x]['inCitations']) for x in all_docs_dict.keys()}
    centralities = {}
    centralities['citations_overall'] = np.vectorize(id2NoCits.get)(ordered_paper_ids)
    print('Done citations_overall centrality', flush=True)
    paper_with_in_cits = citations_df['to'].value_counts().index.values
#     paper_with_cits = list(citations_df['from'].value_counts().index.values) + list(citations_df['to'].value_counts().index.values)
    paper_without_in_cits = set(ordered_paper_ids).difference(set(paper_with_in_cits))
    centralities['in_degree'] = citations_df['to'].value_counts().append(pd.Series(data = np.zeros(len(paper_without_in_cits)),index=paper_without_in_cits)).loc[ordered_paper_ids].values
    print('Done in_degree centrality', flush=True)
    paper_with_out_cits = citations_df['from'].value_counts().index.values
    paper_without_out_cits = set(ordered_paper_ids).difference(set(paper_with_out_cits))
    centralities['out_degree'] = citations_df['from'].value_counts().append(pd.Series(data = np.zeros(len(paper_without_out_cits)),index=paper_without_out_cits)).loc[ordered_paper_ids].values
    print('Done out_degree centrality', flush=True)
    if len(ordered_paper_ids) > 1000:
        centralities['eigenvector'] = graph_tool.centrality.eigenvector(hyperlink_g)[1]._get_data()
        print('Done eigenvector centrality', flush=True)
    centralities['betweenness'] = graph_tool.centrality.betweenness(hyperlink_g)[0]._get_data()
    print('Done betweenness centrality', flush=True)
    centralities['closeness'] = graph_tool.centrality.closeness(hyperlink_g)._get_data()
    print('Done closeness centrality', flush=True)
    centralities['pagerank'] = graph_tool.centrality.pagerank(hyperlink_g)._get_data()
    print('Done pagerank centrality', flush=True)
    centralities['katz'] = graph_tool.centrality.katz(hyperlink_g)._get_data()
    print('Done katz centrality', flush=True)

    with gzip.open(f'{results_folder}hyperlink_g_centralities{filter_label}.pkl.gz','wb') as fp:
        pickle.dump(centralities,fp)
    
    return centralities