# A broad overview of transformer models 

## What are embeddings ? 
Main paper: An Introduction to Transformers (Richard E Turner Microsoft Reseach/ University of Cambridge)
Explanation of how sentence embeddings are generated.
Focus on attention : (paper: Attention is all you need)

## An intuitive understanding of embeddings
How we know that LLMs understand the meaning of words in context. 
Main paper: Visualizing and measuring the geometry of BERT
Presentation of the mathematical techniques used to show a visual representation of "meaning".
How much detail should we have for the maths behind clustering, dimension reduction and euclidian distances ?

## Sentiment analysis in finance
A short history of the literature: from dictionary of sentiments to GPTs & NLP. 
Show where/when analysis of embeddings has been performed in this context. 
-> how much detail ?

# Analyzing embeddings

## Do transformers encode financial sentiment ? 
K-means clustering for financial headlines embeddings. 
Check if clusters replicate the original labels (e.g. graphs in get_graphs.ipynb). (2 dimensiosn + 3 dimensions)
With three clusters and then with two clusters

Possibility to find sentences about the same company / with the same structure ? 
Check cluster percentages in this context 
However mention that results will be less significative because of smaller sample size
Maybe select those that specifically mention numbers (leftmost clusters in phoenix)

Check euclidian distances (after dimension reduction) between embeddings to see if there is a linear transform that could represent an encoding for financial sentiment.
No comparison with regular sentiment. 
Dataset: from FinBERT (AllAgree)

idea: analysis of the embedding of the most common word in the context of a negative sentence vs a positive sentence

## Are daily embeddings linked to markets ?
Problem: find a way to have only one embedding per day... 

Regroup US news: fast text language detection + database of countries to search with regex (if a country is mentioned) -> instead try the MSCI WORLD

Here: regroup sentence embeddings per day, and test the correlation with various market data (VIX, index vol, index returns etc) -> pb: financial headlines might not be linked to a specific market/country/region. -> MSCI world

We can also perform outlier detection and see if it might be correlated with exceptional market/historical events. -> variance of above a certain threshold in the daily embeddings: might not return much because generally speaking embeddings are very close together. 

Also perform analysis with lagged data (1d lag) to see if it would be applicable in real time. Compare the two results. -> NO not possible 
Dataset : Lab1

## Link between the two
Regroup the results from the two experiments, see if they make sense together or if they contradict each other. 
Make sure to show how both results could be biased either way, show vulnerabilities in the datasets, mention ways that the analysis could be improved.

--- 

#### TODO:
- Intuitive explanation  of the following mechanisms: (might cut based on how long each part it) + quick review of the maths
    - Embeddings matrix (initial)
    - Attention 
    - Transformer Block 
This should be the 1st part 
- Presentation of embeddings: show a concrete example of linear transform (code the graphs)
    - 
    
