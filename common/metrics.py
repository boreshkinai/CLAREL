from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np
import logging


def ap_at_k_nn(support_embeddings, query_embeddings, class_ids, k=50):
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
        
    return {'AP@50': np.array(list(ap50.values())).mean(), 'Top-1 Acc': top1_acc}


def get_prototypes(embeddings, labels, vectors_per_protype):
    labels = np.array(labels)
    classes = list(set(labels))
    idxs = np.arange(len(labels))
    prototypes = {}
    for cl in classes:
        class_idxs = idxs[labels == cl]
        class_idxs_protype = np.random.choice(class_idxs, replace=False,
                                              size=min(vectors_per_protype, len(class_idxs)))
        prototype_embeddings = embeddings[class_idxs_protype]
        prototypes[cl] = prototype_embeddings.mean(axis=0)
    return prototypes


def ap_at_k_prototypes(support_embeddings, query_embeddings, class_ids, k=50, num_texts=[1, 2, 3, 5, 10, 20, 50]):
    class_ids = np.array(class_ids)
    print("Compute KD-tree")
    kdtree_query = cKDTree(query_embeddings, balanced_tree=False, compact_nodes=False)
    metrics = {}
    for current_num_texts in num_texts:
        prototypes = get_prototypes(embeddings=support_embeddings, labels=class_ids, 
                                    vectors_per_protype=current_num_texts)
        kdtree_support = cKDTree(np.array(list(prototypes.values())), balanced_tree=False, compact_nodes=False)
        class_ids_support = np.array(list(prototypes.keys()))
    
        # Measure zero-shot classification accuracy
        # for each sample from query set, identify the closest member of support set
        # count the number of times the class of the member identified matches the class of the query
        top1_acc = 0.0
        for query_id, query_embedding in enumerate(tqdm(query_embeddings)):
            actual_class = class_ids[query_id]
            nns, nn_idxs = kdtree_support.query(query_embedding, k=1)
            nearest_class_support = class_ids_support[nn_idxs]
            top1_acc += np.float(nearest_class_support == actual_class)
        # Measure the retrieval AP@50
        ap50matches = {}
        ap50counts = {}
        ap50 = {}
        for class_id_support in tqdm(class_ids_support):
            nns, nn_idxs = kdtree_query.query(prototypes[class_id_support], k=k)
            nearest_classes_query = class_ids[nn_idxs]
            ap50matches[class_id_support] = ap50matches.get(class_id_support, 0.0) + sum(
                [c == class_id_support for c in nearest_classes_query])
            ap50counts[class_id_support] = ap50counts.get(class_id_support, 0.0) + k
        for key in ap50matches.keys():
            ap50[key] = ap50matches[key] / ap50counts[key]

        top1_acc /= len(query_embeddings)
        metrics['AP@%d/#sentences%d'%(k, 10*current_num_texts)] = np.array(list(ap50.values())).mean()
        metrics['Top-1 Acc/#sentences%d'%(10*current_num_texts)] = top1_acc
    
    return metrics

def per_class_average_top1_acc(labels, predictions):
    top1_acc_matches = {}
    top1_acc_counts = {}
    top1_acc = {} 
    for query_id, nearest_class in enumerate(predictions):
        actual_class = labels[query_id]
        top1_acc_matches[actual_class] = top1_acc_matches.get(actual_class, 0.0) + np.float(nearest_class == actual_class)
        top1_acc_counts[actual_class] = top1_acc_counts.get(actual_class, 0.0) + 1

    for key in top1_acc_matches.keys():
        top1_acc[key] = top1_acc_matches[key] / top1_acc_counts[key]
    return np.array(list(top1_acc.values())).mean()

def top1_gzsl(support_embeddings, query_embeddings, class_ids_support, class_ids_query, 
              num_texts=[1, 2, 3, 5, 10, 20, 50], seen_unseen_subsets=None):
    class_ids_query = np.array(class_ids_query)
    class_ids_support = np.array(class_ids_support)
    metrics = {}
    for current_num_texts in tqdm(num_texts):
        prototypes = get_prototypes(embeddings=support_embeddings, labels=class_ids_support, 
                                    vectors_per_protype=current_num_texts)
        class_ids_prototypes = np.array(list(prototypes.keys()))
        array_prototypes = np.array(list(prototypes.values()))
        
        seen_unseen_flag = np.isin(class_ids_prototypes, seen_unseen_subsets['seen']).astype(np.float32)
        
        # Measure generalized zero-shot classification accuracy
        # for each sample from query set, identify the closest member of support set
        # count the number of times the class of the member identified matches the class of the query
        dist = np.sum(np.square(array_prototypes[:,:,None] - query_embeddings.transpose()), axis=1)
        dist = dist * (0.02*seen_unseen_flag[:,None] + 1.0)
        nn_idxs = np.argmin(dist, axis=0)
        top1_acc = per_class_average_top1_acc(labels=class_ids_query, predictions=class_ids_prototypes[nn_idxs])
        
        metrics['Top-1 Acc/#sentences%d'%(10*current_num_texts)] = top1_acc
    return metrics


def top1_gzsl_kdtree(support_embeddings, query_embeddings, class_ids_support, class_ids_query, num_texts=[1, 2, 3, 5, 10, 20, 50]):
    class_ids_query = np.array(class_ids_query)
    class_ids_support = np.array(class_ids_support)
    metrics = {}
    for current_num_texts in tqdm(num_texts):
        prototypes = get_prototypes(embeddings=support_embeddings, labels=class_ids_support, 
                                    vectors_per_protype=current_num_texts)
        kdtree_support = cKDTree(np.array(list(prototypes.values())), balanced_tree=False, compact_nodes=False)
        class_ids_prototypes = np.array(list(prototypes.keys()))
        array_prototypes = np.array(list(prototypes.values()))
        
        # Measure generalized zero-shot classification accuracy
        # for each sample from query set, identify the closest member of support set
        # count the number of times the class of the member identified matches the class of the query
        top1_acc_matches = {}
        top1_acc_counts = {}
        top1_acc = {}
        for query_id, query_embedding in enumerate(tqdm(query_embeddings)):
            actual_class = class_ids_query[query_id]
            nns, nn_idxs = kdtree_support.query(query_embedding, k=1)
            nearest_class = class_ids_prototypes[nn_idxs]
            
            top1_acc_matches[actual_class] = top1_acc_matches.get(actual_class, 0.0) + np.float(nearest_class == actual_class)
            top1_acc_counts[actual_class] = top1_acc_counts.get(actual_class, 0.0) + 1
            
        for key in top1_acc_matches.keys():
            top1_acc[key] = top1_acc_matches[key] / top1_acc_counts[key]
        
        metrics['Top-1 Acc/#sentences%d'%(10*current_num_texts)] = np.array(list(top1_acc.values())).mean()
    return metrics

