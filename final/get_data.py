from tests.api_key import API_KEY
from mistralai.client import MistralClient
import polars as pl 

"""
Functions to import either the labelled data or the dated data 
Adds embeddings with mistral AI 
Generated polars lazyframe with columns : sentences, labels, embeddings OR dates, sentences, embeddings
"""

# import functions 

def import_labelled_data(path: str = r"C:\Users\faune\Downloads\FinancialPhraseBank-v1.0\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt") -> pl.LazyFrame:
    sentences = (
        pl.read_csv(
            source= path,
            separator="\t",
            ignore_errors=True,
            has_header=False
        )
        .lazy()
        .rename({'column_1': 'sentences'})
        .select(pl.col('sentences').str.split('@').list.to_struct()).unnest('sentences')
        .rename({'field_0': 'sentences', 'field_1':'label'})
        .drop_nulls()
    )
    return sentences

def import_dated_data(path: str) -> pl.LazyFrame: #TODO 
    raise NotImplementedError

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

def get_labelled_frame(path: str = r"C:\Users\faune\Downloads\FinancialPhraseBank-v1.0\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt") -> pl.LazyFrame:
    frame = import_labelled_data(path = path)
    return add_embeddings_to_frame(frame=frame)
