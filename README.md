Alignment of comparable corpora by identifying parallel annotations in 
the corpora.

Given a source and target dataset, 
LDA probes for an inter-label matrix containing similarity scores between 
each pair of labels from both the Source and Target dataset.

A label's representation is obtained by computing the centroid of all 
embeddings of entity phrases annotated by the label. 

Cosine distance's between each labels' representation and each of the  
representations of the other labels is computed. A label is aligned with another
if their representations cosine distance is smallest.

