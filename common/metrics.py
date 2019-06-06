from tqdm import tqdm
import numpy as np
import logging


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
    
    
def ap_at_k(labels, predictions, k):
    assert len(labels) == len(predictions)
    # Measure the retrieval AP@50
    ap50matches = {}
    ap50counts = {}
    ap50 = {}
    for idx, label in enumerate(labels):
        nearest_classes_query = predictions[idx][:k]
        ap50matches[label] = ap50matches.get(label, 0.0) + sum(
            [c == label for c in nearest_classes_query])
        ap50counts[label] = ap50counts.get(label, 0.0) + k
    for key in ap50matches.keys():
        ap50[key] = ap50matches[key] / ap50counts[key]
    return np.array(list(ap50.values())).mean()
    
    
def ap_at_k_prototypes(support_embeddings, query_embeddings, class_ids, k=50, 
                       num_texts=[1, 2, 3, 5, 10, 20, 50], distance_metric_prototypes=None):
    class_ids = np.array(class_ids)
    metrics = {}
    for current_num_texts in num_texts:
        prototypes = get_prototypes(embeddings=support_embeddings, labels=class_ids, 
                                    vectors_per_protype=current_num_texts)
        array_prototypes = np.array(list(prototypes.values())) 
        class_ids_support = np.array(list(prototypes.keys()))
        # Measure zero-shot classification accuracy
        # for each sample from query set, identify the closest member of support set
        # count the number of times the class of the member identified matches the class of the query, average over classes
        if distance_metric_prototypes is None:
            dist = np.sqrt(np.sum(np.square(array_prototypes[:,:,None] - query_embeddings.transpose()), axis=1))
        else:
            dist = distance_metric_prototypes.predict_all(query_embeddings, array_prototypes).transpose()
        nn_idxs = np.argmin(dist, axis=0)
        top1_acc = per_class_average_top1_acc(labels=class_ids, predictions=class_ids_support[nn_idxs])
        
        nearest_image_idxs = np.argsort(dist, axis=1)
        nearest_image_classes = class_ids[nearest_image_idxs]
        ap50_val = ap_at_k(labels=class_ids_support, predictions=nearest_image_classes, k=k)

        metrics['AP@%d/#sentences%d'%(k, 10*current_num_texts)] = ap50_val
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
              num_texts=[1, 2, 3, 5, 10, 20, 50], seen_unseen_subsets=None, 
              distance_metric=None, seen_adjustment=0.0):
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
        if distance_metric is None:
            dist = np.sqrt(np.sum(np.square(array_prototypes[:,:,None] - query_embeddings.transpose()), axis=1))
        else:
            dist = distance_metric.predict_all(query_embeddings, array_prototypes).transpose()
        
        seen_unseen_images = np.isin(class_ids_query, seen_unseen_subsets['seen'])
        dist_var = dist*dist        
        
        dist_var_seen = np.sqrt(dist_var[seen_unseen_flag.astype(np.bool)].min(axis=0).mean())
        dist_var_unseen = np.sqrt(dist_var[~seen_unseen_flag.astype(np.bool)].min(axis=0).mean())
        
        dist = dist * (seen_adjustment*seen_unseen_flag[:,None] + 1.0)
        nn_idxs = np.argmin(dist, axis=0)
        top1_acc = per_class_average_top1_acc(labels=class_ids_query, predictions=class_ids_prototypes[nn_idxs])
        
        metrics['Top-1 Acc/#sentences%d'%(10*current_num_texts)] = top1_acc
    return metrics

