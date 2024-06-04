from api_key import API_KEY
from mistralai.client import MistralClient
import polars as pl 

"""
Functions to import either the labelled data or the dated data 
Adds embeddings with mistral AI 
Generated polars lazyframe with columns : sentences, labels, embeddings OR dates, sentences, embeddings
"""


def import_sentences( dated: bool = False, consensus: str = "All", reduce_frame: bool = True) -> pl.LazyFrame:
    """gets the raw formatted data from both data sets

    Args:
        dated (bool, optional): gets the sentences from the dated dataset (origin: lab1). Defaults to False.
        consensus (str, optional): consensus for the labels on the 2nd dataset (undated, used to train finBERT). Defaults to "All".
        Can be either "All", "75", "66" or "50". Don't forget to use a string data type even if you use the numbered options.
        reduce_frame (bool, optional): reduces size of the frame from 1m + to ~20 000. Defaults to True.
        (only data after 2011 + only euro characters + divides the nb of sentences per day by 40)

    Returns:
        pl.LazyFrame: lazyframe with 'sentences' and 'label' columns (and also 'date' if dated = True) 
    """    
    if dated:
        sentences = (
            pl.read_csv(r"C:\Users\faune\Downloads\lab1\lab1\data\dow_jones_news.csv", separator=';')
            .with_columns(pl.col('news').str.split('***'))
            .explode('news')
            .group_by('Date').agg(pl.col('news'), pl.col('Label').first())
            .sort('Date')
            .rename({'Date': 'date', 'Label': 'label', 'news': 'sentences'})
            .with_columns(pl.col('label').replace({0:'negative', 1:'positive'}), pl.col('date').cast(pl.Date))
            .explode('sentences')
            .lazy()
        )
        if reduce_frame:
            sentences = (
                sentences
                .collect()
                .filter(pl.col('date').dt.year().gt(2010), pl.col('sentences').str.contains(pattern = '[a-zA-Z]+'))
                .group_by('date')
                .agg(pl.col('sentences'), pl.col('label').first())
                .with_columns(pl.col('sentences').list.gather_every(40))
                .explode('sentences')
                .sort('date')
                .lazy()
            )
    else:
        sentences = (
            pl.read_csv(
                source= fr"C:\Users\faune\Downloads\FinancialPhraseBank-v1.0\FinancialPhraseBank-v1.0\Sentences_{consensus}Agree.txt",
                separator="\t",
                ignore_errors=True,
                has_header=False
            )
            .rename({'column_1': 'sentences'})
            .select(pl.col('sentences').str.split('@').list.to_struct()).unnest('sentences')
            .rename({'field_0': 'sentences', 'field_1':'label'})
            .drop_nulls()
            .unique()
            .lazy()
        )
    return sentences

def _get_embeddings_by_chunks(data: list, chunk_size: int, api_key: str = API_KEY): # mostly COPIED FROM https://docs.mistral.ai/capabilities/embeddings/
    client = MistralClient(api_key=api_key)
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client.embeddings(model="mistral-embed", input=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data] 

def get_embeddings(dated: bool = False, consensus: str = 'All', reduce_frame: bool = True) -> pl.LazyFrame:
    """gets the data along with an "embeddings" column

    Args:
        dated (bool, optional): gets the sentences from the dated dataset (origin: lab1). Defaults to False.
        consensus (str, optional): consensus for the labels on the 2nd dataset (undated, used to train finBERT). Defaults to "All".
        Can be either "All", "75", "66" or "50". Don't forget to use a string data type even if you use the numbered options.
        reduce_frame (bool, optional): reduces size of the frame from 1m + to ~20 000. Defaults to True.
        (only data after 2011 + only euro characters + divides the nb of sentences per day by 40)

    Returns:
        pl.LazyFrame: lazyframe with 'sentences', 'label' and 'embeddings' columns (and also 'date' if dated = True) 
    """    
    frame = import_sentences(dated=dated, consensus=consensus, reduce_frame=reduce_frame)
    return (
        frame
        .with_columns(
            pl.Series(
                name= 'embeddings', 
                values= _get_embeddings_by_chunks(frame.select('sentences').collect().to_series().to_list(), 50)
            )
        )
        .lazy()
    )