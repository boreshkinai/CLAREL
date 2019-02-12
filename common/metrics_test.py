import numpy as np
from common.metrics import ap_at_k_prototypes


NSAMPLES = 2500
NCLASSES = 50
EMBED_DIM = 1024

# Test the case of random embeddings
support_embeddings = np.random.normal(size=(NSAMPLES, EMBED_DIM))
query_embeddings = np.random.normal(size=(NSAMPLES, EMBED_DIM))
class_ids = np.random.randint(low=0, high=NCLASSES, size=NSAMPLES)

# We expect metrics to be centered around 2% (1/50 classes)
metrics = ap_at_k_prototypes(support_embeddings, query_embeddings, class_ids, k=50, num_texts=[1, 2, 3, 5, 10, 20])
print(metrics)


# Test the case when 75% of embeddings are close
for i in range(NCLASSES):
    mean_class = i*10
    std_class = 20.0

    idxs = class_ids == i
    indicators = np.asanyarray(np.random.uniform(size=(sum(idxs), 1)) > 0.25, np.float32)
    support_embeddings[idxs] = mean_class + std_class * support_embeddings[idxs]
    query_embeddings[idxs] = mean_class*indicators + std_class * query_embeddings[idxs]

# We expect metrics approach 75% as the num_texts grows
metrics = ap_at_k_prototypes(support_embeddings, query_embeddings, class_ids, k=50, num_texts=[1, 2, 3, 5, 10, 20])
print(metrics)








