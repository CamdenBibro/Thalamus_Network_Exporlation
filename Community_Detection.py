
# Community Detection (Exploration and Testing Script)
# This script is used to explore community detection algorithms and test them on the data. 
# I am interested in the community behavior of the thalamus and cortex regions.

# Camden Bibro
# 2024-11-04

import numpy as np
import community as community_louvain  # community-louvain package
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Load in data cleaned from the previous notebook

cons_ctx_data = np.load(r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/py_data_without_pickle/cons_ctx_data.npy')
pats_ctx_data = np.load(r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/py_data_without_pickle/pats_ctx_data.npy')
ctx_regions = np.load(r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/py_data_without_pickle/ctx_regions.npy', allow_pickle=True)


def calculate_nmi_for_pairs(partitions):
    nmi_scores = [
        normalized_mutual_info_score(partitions[i], partitions[j])
        for i in range(len(partitions))
        for j in range(i + 1, len(partitions))
    ]
    return nmi_scores


def zscore_data_ignoring_zeros(data): # Use zscore for ML techniques preserve distribution
    # Make sure all NaNs are zeros
    
    # Create a boolean mask to identify non-zero elements
    non_zero_mask = data != 0
    zscored_data = np.zeros_like(data)
    
    # Loop over each subject's matrix to z-score individually, ignoring zeros
    for i in range(data.shape[2]):
        # Extract the non-zero elements for the current matrix
        non_zero_values = data[:, :, i][non_zero_mask[:, :, i]]
        
        # Calculate mean and std only for non-zero elements
        mean_non_zero = np.mean(non_zero_values)
        std_non_zero = np.std(non_zero_values)
        
        # Z-score only the non-zero elements
        zscored_data[:, :, i][non_zero_mask[:, :, i]] = (data[:, :, i][non_zero_mask[:, :, i]] - mean_non_zero) / std_non_zero

    return zscored_data


def fisher_z_transform_ignoring_zeros(data): # Use fisher z-transform for correlation data significance testing 
    # Create a boolean mask to identify non-zero elements
    non_zero_mask = data != 0
    fisher_z_data = np.zeros_like(data)

    # Loop over each subject's matrix to apply Fisher z-transformation individually, ignoring zeros
    for i in range(data.shape[2]):
        # Apply Fisher z-transformation only to the non-zero elements
        fisher_z_data[:, :, i][non_zero_mask[:, :, i]] = 0.5 * np.log((1 + data[:, :, i][non_zero_mask[:, :, i]]) / 
                                                                      (1 - data[:, :, i][non_zero_mask[:, :, i]]))

    return fisher_z_data


def consensus_louvain(data, gamma, iterations=1000):
    # Store all partitions
    partitions = []
    G = nx.from_numpy_array(data)  # Create a NetworkX graph from adjacency matrix
    
    for _ in range(iterations):
        partition = community_louvain.best_partition(G, resolution=gamma)
        partitions.append(list(partition.values()))
    
    # Continue with consensus calculations as before
    consensus_matrix = np.zeros((data.shape[0], data.shape[0]))
    
    for partition in partitions:
        for i in range(len(partition)):
            for j in range(len(partition)):
                if partition[i] == partition[j]:  # Same community
                    consensus_matrix[i, j] += 1
    
    # Normalize consensus matrix
    consensus_matrix /= iterations
    return partitions, consensus_matrix


#%%

# Combine patient and control data for community detection
ctx_data = np.concatenate((cons_ctx_data, pats_ctx_data), axis=2)

# Z-score the data for ML techniques community louvain
ctx_data = zscore_data_ignoring_zeros(ctx_data)
# Change all negative values to zero
ctx_data[ctx_data < 0] = 0 

# Histogram of z-scored data 
plt.hist(ctx_data.flatten(), bins=200, color='skyblue', edgecolor='black')

# Make plan to sweep over gamma values!
# Sweep over gamma values
#gamma_values = np.arange(0.5, 2.1, 0.1)
#mean_nmi_scores = []

gamma1 = 1.5
gamma2 = 1.2


final_partitions = np.zeros((ctx_data.shape[2], ctx_data.shape[0]))
final_consensus_matrix = np.zeros((ctx_data.shape[0], ctx_data.shape[0]))

for subj in range(ctx_data.shape[2]):
    
    X = ctx_data[:,:,subj]
    # set diagonal to zero
    np.fill_diagonal(X, 0)
    partitions,_ = consensus_louvain(X, gamma1, iterations=200)
    consensus_matrix = np.zeros((ctx_data.shape[0], ctx_data.shape[0]))
    
    for i in range(len(partitions)):
        # take the i value of partitions and make it a numpy array
        P = np.array(partitions[i])
        K = np.equal(P[:, np.newaxis], P)
        consensus_matrix += K 
    
    consensus_matrix /= len(partitions)
    np.fill_diagonal(consensus_matrix, 0)
    G_consensus = nx.from_numpy_array(consensus_matrix)  # Create a graph from the consensus matrix        
    
    # Find communities from the consensus matrix for one subject
    second_partition = community_louvain.best_partition(G_consensus, resolution=gamma2)
    # convert the final partitions to a numpy array
    final_partitions[subj,:] = np.array(list(second_partition.values()))

    print(f"Subject {subj} done")

for i in range(final_partitions.shape[0]):
    P2 = final_partitions[i,:]
    K2 = np.equal(P2[:, np.newaxis], P2)
    final_consensus_matrix += K2    
    
final_consensus_matrix /= final_partitions.shape[0]
np.fill_diagonal(final_consensus_matrix, 0)
# Threshold the final consensus matrix as numpy array
tmp = np.array(final_consensus_matrix)
tmp[tmp > 0.75] = 0
# convert to graph
G_final = nx.from_numpy_array(tmp)
# Find communities from the final consensus matrix
final_partition = community_louvain.best_partition(G_final, resolution=gamma2)
# convert the final partitions to a numpy array
community_structure = np.array(list(final_partition.values()))


# Really just doing this for plotting purposes! 
# Only showing the largest positive connections. 
tmp[tmp < 0.40] = 0
G_final = nx.from_numpy_array(tmp)


#%% GRAPH and PLOT

# Visualize positive connections community structure
plt.figure(figsize=(12, 6))
plt.subplot(121)
nx.draw(G_final, pos=nx.spring_layout(G_final), 
        node_color=list(community_structure), cmap=plt.cm.viridis, 
        node_size=50, edge_color="lightgray", alpha=0.7, labels=True)
plt.title("Positive Connections")






