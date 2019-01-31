from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np


def ap_at_k(support_embeddings, query_embeddings, class_ids, k=50):
    class_ids = np.array(class_ids)
    kdtree = cKDTree(support_embeddings)
    ap50matches = {}
    ap50counts = {}
    ap50 = {}
    top1_acc=0
    for query_id, query_embedding in enumerate(tqdm(query_embeddings)):
        actual_class = class_ids[query_id]
        nns, nn_idxs = kdtree.query(query_embedding, k=k)
        nn_idxs = nn_idxs.tolist()

        nearest_classes = class_ids[nn_idxs]

        ap50matches[actual_class] = ap50matches.get(actual_class, 0.0) + sum([c == actual_class for c in nearest_classes])
        ap50counts[actual_class] = ap50counts.get(actual_class, 0.0) + k
        
        top1_acc += np.float(nearest_classes[0] == actual_class)

    for key in ap50matches.keys():
        ap50[key] = ap50matches[key] / ap50counts[key]
        
    top1_acc /= len(query_embeddings)
        
    return np.array(list(ap50.values())).mean(), top1_acc