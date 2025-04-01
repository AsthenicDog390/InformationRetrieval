import pyterrier as pt
import pandas as pd

if not pt.started():
    pt.init()

def rewrite_queries_RM3(df: pd.DataFrame, index_ref, fb_terms: int = 10, fb_docs: int = 5, orig_weight: float = 0.5) -> pd.DataFrame:
    """
    Rewrites a dataframe of queries using RM3 expansion.
    
    :param df: A pandas DataFrame with a column 'query' containing queries.
    :param index_path: The path to the Terrier index.
    :param fb_terms: Number of feedback terms to add.
    :param fb_docs: Number of feedback documents to consider.
    :param orig_weight: Weight of the original query terms.
    :return: DataFrame with an additional column 'expanded_query'.
    """
    # Make sure the DataFrame has the required columns for PyTerrier
    if 'qid' not in df.columns:
        df = df.copy()
        df['qid'] = [str(i) for i in range(len(df))]
    
    # Create a retrieval and rewrite pipeline
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    rm3 = pt.rewrite.RM3(index_ref)
    
    # Process each query individually to get the expanded version
    expanded_queries = df.copy()
    expanded_queries['expanded_query'] = ""
    
    for idx, row in df.iterrows():
        # Create a single query dataframe
        query_df = pd.DataFrame([{'qid': row['qid'], 'query': row['query']}])
        
        # First retrieve documents with BM25
        results = bm25.transform(query_df)
        
        # Then expand the query with RM3
        expanded_query = rm3.transform(results)
        
        # Store the expanded query
        if not expanded_query.empty:
            expanded_queries.at[idx, 'expanded_query'] = expanded_query.iloc[0]['query']
        else:
            # Keep the original if expansion fails
            expanded_queries.at[idx, 'expanded_query'] = row['query']
    
    return expanded_queries

def rewrite_queries_BO1(df, index_ref):
    """
    Rewrites a dataframe of queries using BO1 expansion.
    
    :param df: A pandas DataFrame with a column 'query' containing queries.
    :param index_ref: The path to the Terrier index.
    :return: DataFrame with an additional column 'expanded_query'.
    """
    # Make sure the DataFrame has the required columns for PyTerrier
    if 'qid' not in df.columns:
        df = df.copy()
        df['qid'] = [str(i) for i in range(len(df))]
    
    # Create a retrieval and rewrite pipeline
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    bo1 = pt.rewrite.Bo1QueryExpansion(index_ref)
    
    # Process each query individually to get the expanded version
    expanded_queries = df.copy()
    expanded_queries['expanded_query'] = ""
    
    for idx, row in df.iterrows():
        # Create a single query dataframe
        query_df = pd.DataFrame([{'qid': row['qid'], 'query': row['query']}])
        
        # First retrieve documents with BM25
        results = bm25.transform(query_df)
        
        # Then expand the query with BO1
        expanded_query = bo1.transform(results)
        
        # Store the expanded query
        if not expanded_query.empty:
            expanded_queries.at[idx, 'expanded_query'] = expanded_query.iloc[0]['query']
        else:
            # Keep the original if expansion fails
            expanded_queries.at[idx, 'expanded_query'] = row['query']
    
    return expanded_queries

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Ensure NLTK stopwords are downloaded
import nltk
# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')  # Open Multilingual WordNet

def rewrite_queries_using_porter_stemmer_and_stop_words(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rewrites a dataframe of queries using Porter Stemmer and removes stopwords.
    
    :param df: A pandas DataFrame with a column 'query' containing queries.
    :return: DataFrame with an additional column 'expanded_query'.
    """
    # Use NLTK's English stopwords
    stopwords_set = set(stopwords.words('english'))
    
    # Initialize NLTK's Porter Stemmer
    stemmer = PorterStemmer()

    # Make sure the DataFrame has the required columns for PyTerrier
    if 'qid' not in df.columns:
        df = df.copy()
        df['qid'] = [str(i) for i in range(len(df))]
    
    # Process each query individually to get the expanded version
    expanded_queries = df.copy()
    expanded_queries['expanded_query'] = ""
    
    for idx, row in df.iterrows():
        # Create a single query dataframe
        query_df = pd.DataFrame([{'qid': row['qid'], 'query': row['query']}])

        # Tokenize the query
        tokens = query_df['query'].iloc[0].split()

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stopwords_set]

        # Apply stemming
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Join the tokens back into a string
        expanded_query = ' '.join(stemmed_tokens)

        query_df['expanded_query'] = expanded_query
        # Store the expanded query
        expanded_queries.at[idx, 'expanded_query'] = expanded_query
    
    return expanded_queries

def rewrite_queries_using_snowball_stemmer_and_stop_words(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rewrites a dataframe of queries using Snowball Stemmer and removes stopwords.
    
    :param df: A pandas DataFrame with a column 'query' containing queries.
    :return: DataFrame with an additional column 'expanded_query'.
    """
    # Use NLTK's English stopwords
    stopwords_set = set(stopwords.words('english'))
    
    # Initialize NLTK's Snowball Stemmer
    stemmer = SnowballStemmer("english")

    # Make sure the DataFrame has the required columns for PyTerrier
    if 'qid' not in df.columns:
        df = df.copy()
        df['qid'] = [str(i) for i in range(len(df))]
    
    # Process each query individually to get the expanded version
    expanded_queries = df.copy()
    expanded_queries['expanded_query'] = ""
    
    for idx, row in df.iterrows():
        # Create a single query dataframe
        query_df = pd.DataFrame([{'qid': row['qid'], 'query': row['query']}])

        # Tokenize the query
        tokens = query_df['query'].iloc[0].split()

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stopwords_set]

        # Apply stemming
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Join the tokens back into a string
        expanded_query = ' '.join(stemmed_tokens)

        query_df['expanded_query'] = expanded_query
        # Store the expanded query
        expanded_queries.at[idx, 'expanded_query'] = expanded_query
    
    return expanded_queries

def rewrite_queries_using_lemmatization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rewrites a dataframe of queries using WordNet Lemmatizer and removes stopwords.
    
    :param df: A pandas DataFrame with a column 'query' containing queries.
    :return: DataFrame with an additional column 'expanded_query'.
    """
    # Use NLTK's English stopwords
    stopwords_set = set(stopwords.words('english'))
    
    # Initialize NLTK's Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Make sure the DataFrame has the required columns for PyTerrier
    if 'qid' not in df.columns:
        df = df.copy()
        df['qid'] = [str(i) for i in range(len(df))]
    
    # Process each query individually to get the expanded version
    expanded_queries = df.copy()
    expanded_queries['expanded_query'] = ""
    
    for idx, row in df.iterrows():
        # Tokenize the query
        tokens = row['query'].split()

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stopwords_set]

        # Apply lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

        # Join the tokens back into a string
        expanded_query = ' '.join(lemmatized_tokens)

        # Store the expanded query
        expanded_queries.at[idx, 'expanded_query'] = expanded_query
    
    return expanded_queries