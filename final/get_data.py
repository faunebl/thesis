from api_key import API_KEY
from mistralai.client import MistralClient
import polars as pl 

"""
Functions to import either the labelled data or the dated data 
Adds embeddings with mistral AI 
Generated polars lazyframe with columns : sentences, labels, embeddings OR dates, sentences, embeddings
"""

# import functions 

def import_labelled_data(t: str = "All") -> pl.LazyFrame:
    sentences = (
        pl.read_csv(
            source= fr"C:\Users\faune\Downloads\FinancialPhraseBank-v1.0\FinancialPhraseBank-v1.0\Sentences_{t}Agree.txt",
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

def import_dated_data(path: str = r"C:\Users\faune\Downloads\lab1\lab1\data\dow_jones_news.csv") -> pl.LazyFrame: #TODO 
    return pl.read_csv(path)
# getting embeddings 

def _get_embeddings_by_chunks(data: list, chunk_size: int, api_key: str = API_KEY): # mostly COPIED FROM https://docs.mistral.ai/capabilities/embeddings/
    client = MistralClient(api_key=api_key)
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client.embeddings(model="mistral-embed", input=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data]

def add_embeddings_to_frame(frame: pl.LazyFrame):
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

# synthactic sugar 

def get_labelled_frame(t: str = 'All') -> pl.LazyFrame:
    frame = import_labelled_data(t = t)
    return add_embeddings_to_frame(frame=frame)
