#!/usr/bin/env python
# coding: utf-8

# In[2]:


#relevant libraries
import numpy as np
import urllib.request
import os
import matplotlib.pyplot as plt
import sklearn.linear_model as sk


# In[3]:


#preprocessing
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
file_path = "mnist.npz"

if not os.path.exists(file_path):
    urllib.request.urlretrieve(url, file_path)

with np.load(file_path) as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

train_mask = np.isin(y_train, [0, 1, 2])
test_mask = np.isin(y_test, [0, 1, 2])

x_train_filtered = x_train[train_mask]
y_train_filtered = y_train[train_mask]

x_test_filtered = x_test[test_mask]
y_test_filtered = y_test[test_mask]

x_train_flat = x_train_filtered.reshape(x_train_filtered.shape[0], -1)
x_test_flat = x_test_filtered.reshape(x_test_filtered.shape[0], -1)

x_train_processed = x_train_flat.astype(np.float32) / 255.0
x_test_processed = x_test_flat.astype(np.float32) / 255.0

print(f"Original Training set shape: {x_train.shape}, {y_train.shape}")
print(f"Original Test set shape: {x_test.shape}, {y_test.shape}")
print()
print(f"Filtered Training set shape (classes 0, 1, 2): {x_train_filtered.shape}, {y_train_filtered.shape}")
print(f"Filtered Test set shape (classes 0, 1, 2): {x_test_filtered.shape}, {y_test_filtered.shape}")
print()
print(f"Processed Training set shape (flattened): {x_train_processed.shape}")
print(f"Processed Test set shape (flattened): {x_test_processed.shape}")
print(f"Pixel value range: [{x_train_processed.min()}, {x_train_processed.max()}]")


# In[4]:


def compute_mean(data_matrix):
    return np.mean(data_matrix, axis=0)

def compute_covariance(data_matrix):
    N = data_matrix.shape[0]
    return (data_matrix.T @ data_matrix) / (N - 1)
    
def pca(samples, variance=1.0, components=0):
    print(f"Training set shape ({samples.shape[0]},{samples.shape[1]})")
    
    # Calculate mean and center data
    mean = compute_mean(samples)
    print(f"Mean shape {mean.shape}")
    centered_data_mat = samples - mean
    print(f"Centered data matrix shape {centered_data_mat.shape}")
    
    # Compute covariance matrix
    covariance = compute_covariance(centered_data_mat)
    print(f"Centered data matrix shape {covariance.shape}")
    
    # Solve eigenvalue problem (eigh is optimal for symmetric covariance matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Vectorized cleanup of negative eigenvalues caused by computational float errors
    eigenvalues = np.maximum(eigenvalues, 0)
            
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    # Variance Preservation Threshold
    if 0 < variance < 1:
        explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        retain_indices = np.argmax(cumulative_variance >= variance) + 1
        return sorted_eigenvectors[:, :retain_indices]

    # Fixed Number of Components
    elif 0 < components < sorted_eigenvectors.shape[1]:
        return sorted_eigenvectors[:, :components]

    return sorted_eigenvectors

def apply_pca(samples, w, mean):
    samples = samples - mean
    return samples @ w


# In[111]:


#apply pca on data
w_pca = pca(x_train_processed, components=10)
reduced_dim_x_train = apply_pca(x_train_processed, w_pca, np.mean(x_train_processed, axis = 0))
reduced_dim_x_test = apply_pca(x_test_processed, w_pca, np.mean(x_train_processed, axis = 0))


# In[113]:


def gini(y):
    #compute gini(s) as per assignment doc
    if (len(y) <= 0):
        return 0

    labels = {}
    for i in y:
        if (i in labels):
            labels[i]+=1
        else:
            labels[i]=1
            
    pks = np.array([labels[i]/len(y) for i in labels])
    return 1 - np.sum(pks**2)

def weighted_gini_index(left, right):
    #compute weighted gini index
    ll = len(left)
    lr = len(right)
    lt = ll+lr
    if (lt <= 0):
        return 0

    return (ll/lt)*gini(left) + (lr/lt)*gini(right)

def best_split(X_train, y_train):
    #find best split for a node using all features
    split = {}
    best_gini = float('inf')
    
    for feature in range(X_train.shape[0]):
        #split at median value
        threshold = np.median(X_train[:,feature])

        left = y_train[X_train[:,feature] <= threshold]
        right = y_train[X_train[:,feature] > threshold]

        gini = weighted_gini_index(left, right)
        if gini < best_gini:
            best_gini = gini
            split = {'gini':gini, 'feature': feature, 'threshold': threshold}
            
        return split


# In[117]:


#3 leaf decision tree 

def label(y):
    vals, counts = np.unique(y, return_counts=True)
    mode = vals[np.argmax(counts)]
    return mode
    
def decision_tree(X_train, y_train):
    tree = {}
    
    #root split
    split_root = best_split(X_train, y_train)
    tree['root'] = split_root

    #children split
    root_split_feature = split_root['feature']
    #print(root_split_feature)
    root_split_threshold = split_root['threshold']
    #print(root_split_threshold)
    
    left = X_train[:,root_split_feature] <= root_split_threshold
    right = X_train[:,root_split_feature] > root_split_threshold
    
    left_split = best_split(X_train[left], y_train[left])
    right_split = best_split(X_train[right], y_train[right])
    y_left = y_train[left]
    y_right = y_train[right]
    m_total = len(y_left)+len(y_right)

    if ((len(y_left)/m_total) * left_split['gini'] + (len(y_right)/m_total) * gini(y_right) < (len(y_left)/m_total) * gini(y_left) + (len(y_right)/m_total) * right_split['gini']
):
        tree['split_node'] = 'left'
        tree['child_split'] = left_split
        l = X_train[left][:, left_split['feature']] <= left_split['threshold']
        r = X_train[left][:, left_split['feature']] > left_split['threshold']
        tree['leaves'] = {'LL':label(y_train[left][l]),
                          'LR':label(y_train[left][r]),
                          'R':label(y_train[right])}
    else:
        tree['split_node'] = 'right'
        tree['child_split'] = right_split
        l = X_train[right][:, right_split['feature']] <= right_split['threshold']
        r = X_train[right][:, right_split['feature']] > right_split['threshold']
        tree['leaves'] = {'L':label(y_train[left]),
                          'RL':label(y_train[right][l]),
                          'RR':label(y_train[right][r])}

    return tree

def predict(tree, X_test):
    predictions = []
    
    root_feat = tree['root']['feature']
    root_thresh = tree['root']['threshold']
    
    for sample in X_test:
        if sample[root_feat] <= root_thresh:
            if tree['split_node'] == 'left':
                child_feat = tree['child_split']['feature']
                child_thresh = tree['child_split']['threshold']
                
                if sample[child_feat] <= child_thresh:
                    predictions.append(tree['leaves']['LL'])
                else:
                    predictions.append(tree['leaves']['LR'])
            else:
                predictions.append(tree['leaves']['L'])
                
        else:
            if tree['split_node'] == 'right':
                child_feat = tree['child_split']['feature']
                child_thresh = tree['child_split']['threshold']
                
                if sample[child_feat] <= child_thresh:
                    predictions.append(tree['leaves']['RL'])
                else:
                    predictions.append(tree['leaves']['RR'])
            else:
                predictions.append(tree['leaves']['R'])
                
    return np.array(predictions)

def accuracy(y_pred, y_test):
    overall_accuracy = np.mean(y_pred == y_test)
    print(f"Overall Classification Accuracy: {overall_accuracy * 100}%")
    
    print("\nClass-wise Accuracy:")
    for c in [0, 1, 2]:
        class_indices = (y_test == c)
        class_correct = np.sum(y_pred[class_indices] == y_test[class_indices])
        class_total = np.sum(class_indices)
        
        acc = class_correct / class_total
        print(f"Class {c}: {acc * 100}%")


# In[118]:


single_tree = decision_tree(reduced_dim_x_train, y_train_filtered)
print("DECISION TREE")
print(single_tree)

#predictions for test and train sets
y_pred_train = predict(single_tree, reduced_dim_x_train)
y_pred_test = predict(single_tree, reduced_dim_x_test)

#accuracy
print("\nTRAIN SET ACCURACY")
accuracy(y_pred_train, y_train_filtered)
print("\nTEST SET ACCURACY")
accuracy(y_pred_test, y_test_filtered)


# In[178]:


def majority(predictions):
    voted_predictions = []
    
    for i in range(predictions.shape[1]):
        # Get predictions for the i-th sample across all 5 trees
        sample_preds = predictions[:, i]
        vals, counts = np.unique(sample_preds, return_counts=True)
        majority_label = vals[np.argmax(counts)]
        voted_predictions.append(majority_label)
        
    return np.array(voted_predictions)

def bagging_forest_model(X_train, y_train, X_test, y_test, num_trees=5):
    m_total = X_train.shape[0]
    forest = []
    oob_errors = []

    for i in range(num_trees):
        bootstrap_indices = np.random.choice(m_total, size=m_total, replace=True)
        
        X_boot = X_train[bootstrap_indices]
        y_boot = y_train[bootstrap_indices]
        
        all_indices = np.arange(m_total)
        oob_indices = np.setdiff1d(all_indices, bootstrap_indices)
        
        X_oob = X_train[oob_indices]
        y_oob = y_train[oob_indices]
        
        tree = decision_tree(X_boot, y_boot)
        forest.append(tree)
    
        if len(oob_indices) > 0:
            y_oob_pred = predict(tree, X_oob)
            oob_accuracy = np.mean(y_oob_pred == y_oob)
            oob_error = 1.0 - oob_accuracy
            oob_errors.append(oob_error)
        else:
            oob_errors.append(0)

    avg_oob_error = np.mean(oob_errors)
    print(f"\nAverage Out-of-Bag (OOB) Error: {avg_oob_error}")
    #print(len(forest))
    return forest

def predict_forest(forest, X_test):
    all_tree_predictions = []
    for tree in forest:
        preds = predict(tree, X_test)
        all_tree_predictions.append(preds)
        
    all_tree_predictions = np.array(all_tree_predictions)
    
    final_predictions = majority(all_tree_predictions)
    
    return final_predictions 


# In[179]:


bagging_forest = bagging_forest_model(reduced_dim_x_train, y_train_filtered, reduced_dim_x_test, y_test_filtered)
print("\nBAGGING FOREST")
print(bagging_forest)

#predictions for test and train sets
y_pred_train = predict_forest(bagging_forest, reduced_dim_x_train)
y_pred_test = predict_forest(bagging_forest, reduced_dim_x_test)

#accuracy
print("\nTRAIN SET ACCURACY")
accuracy(y_pred_train, y_train_filtered)
print("\nTEST SET ACCURACY")
accuracy(y_pred_test, y_test_filtered)


# In[69]:


def best_split_rf(X, y, k):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    
    n_features = X.shape[1]
    
    #randomly select any k features for split
    feature_indices = np.random.choice(n_features, size=k, replace=False)
    
    for feature_idx in feature_indices:
        threshold = np.median(X[:, feature_idx])
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        if len(y_left) == 0 or len(y_right) == 0:
            continue
            
        current_gini = weighted_gini_index(y_left, y_right)
        
        if current_gini < best_gini:
            best_gini = current_gini
            best_feature = feature_idx
            best_threshold = threshold
            
    return {'gini': best_gini, 'feature': best_feature, 'threshold': best_threshold}


# In[139]:


def decision_tree_rf(X_train, y_train, k):
    tree = {}
    m_total = len(y_train)

    # 1. Root split (passing k)
    split_root = best_split_rf(X_train, y_train, k)
    tree['root'] = split_root

    root_split_feature = split_root['feature']
    root_split_threshold = split_root['threshold']
    
    left = X_train[:, root_split_feature] <= root_split_threshold
    right = X_train[:, root_split_feature] > root_split_threshold
    
    y_left = y_train[left]
    y_right = y_train[right]
    
    left_split = best_split_rf(X_train[left], y_left, k)
    right_split = best_split_rf(X_train[right], y_right, k)

    gini_if_left = (len(y_left)/m_total) * left_split['gini'] + (len(y_right)/m_total) * gini(y_right)
    gini_if_right = (len(y_left)/m_total) * gini(y_left) + (len(y_right)/m_total) * right_split['gini']

    if gini_if_left < gini_if_right:
        tree['split_node'] = 'left'
        tree['child_split'] = left_split
        
        l = X_train[left][:, left_split['feature']] <= left_split['threshold']
        r = X_train[left][:, left_split['feature']] > left_split['threshold']
        
        tree['leaves'] = {
            'LL': label(y_left[l]), 
            'LR': label(y_left[r]), 
            'R':  label(y_right)
        }
    else:
        tree['split_node'] = 'right'
        tree['child_split'] = right_split
        
        l = X_train[right][:, right_split['feature']] <= right_split['threshold']
        r = X_train[right][:, right_split['feature']] > right_split['threshold']
        
        tree['leaves'] = {
            'L':  label(y_left),
            'RL': label(y_right[l]), 
            'RR': label(y_right[r])
        }

    return tree


# In[180]:


def random_forest_model(X_train, y_train, X_test, y_test, num_trees=5, k=3):
    m_total = X_train.shape[0]
    forest = []
    oob_errors = []

    for i in range(num_trees):
        #sample with replacement for bootstrapped dataset
        bootstrap_indices = np.random.choice(m_total, size=m_total, replace=True)
        X_boot, y_boot = X_train[bootstrap_indices], y_train[bootstrap_indices]
        
        #find out out of bag samples
        all_indices = np.arange(m_total)
        oob_indices = np.setdiff1d(all_indices, bootstrap_indices)
        X_oob, y_oob = X_train[oob_indices], y_train[oob_indices]
        
        #train random tree
        tree = decision_tree_rf(X_boot, y_boot, k)
        forest.append(tree)
        
        #oob error 
        if len(oob_indices) > 0:
            y_oob_pred = predict(tree, X_oob)
            oob_error = 1.0 - np.mean(y_oob_pred == y_oob)
            oob_errors.append(oob_error)

    avg_oob_error = np.mean(oob_errors)
    print(f"\nAverage Out-of-Bag (OOB) Error: {avg_oob_error}")
    
    return forest


# In[188]:


random_forest = random_forest_model(reduced_dim_x_train, y_train_filtered, reduced_dim_x_test, y_test_filtered)
print("\nRANDOM FOREST")
print(bagging_forest)

#predictions for test and train sets
y_pred_train = predict_forest(random_forest, reduced_dim_x_train)
y_pred_test = predict_forest(random_forest, reduced_dim_x_test)

#accuracy
print("\nTRAIN SET ACCURACY")
accuracy(y_pred_train, y_train_filtered)
print("\nTEST SET ACCURACY")
accuracy(y_pred_test, y_test_filtered)


# In[ ]:




