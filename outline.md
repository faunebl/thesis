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
Check if clusters replicate the original labels (e.g. graphs in get_graphs.ipynb).
Check euclidian distances (after dimension reduction) between embeddings to see if there is a linear transform that could represent an encoding for financial sentiment.
Should we compare results with regular sentiment ? (i.e. check euclidian distance between two identical sentences with opposite sentiment, such as "I hate my cat" vs. "I love my cat") (might not be possible because no database currently).

## Are daily embeddings linked to markets ?
Here: regroup sentence embeddings per day, and test the correlation with various market data (VIX, index vol, index returns etc) -> pb: financial headlines might not be linked to a specific market/country/region. 
We can also perform outlier detection and see if it might be correlated with exceptional market/historical events.
Also perform analysis with lagged data (1d lag) to see if it would be applicable in real time. Compare the two results. 

## Link between the two
Regroup the results from the two experiments, see if they make sense together or if they contradict each other. 
Make sure to show how both results could be biased either way, show vulnerabilities in the datasets, mention ways that the analysis could be improved.