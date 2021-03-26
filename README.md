**Label-document-Alignment**

Alignment of comparable corpora by identifying parallel annotations in 
the corpora. Label-document-Alignment can be used to address concerns such as
augmenting small datasets or even reducing noise in weakly supervised datasets

**How it works**
Given a source and target dataset, 
L-D-A probes for an inter-label matrix containing similarity scores between 
each pair of labels from both the Source and Target datasets.

A label's representation is obtained by computing the centroid of all 
embeddings of entity phrases annotated by the label. 

Cosine distance's between each labels' representation and each of the  
representations of the other labels is computed. Outcomes annotated with
a particular label are aligned with outcomes for another if the
cosine distance between the label representationss is smallest.

