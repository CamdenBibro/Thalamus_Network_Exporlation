
## Thalamus Connectivity (Network) Analysis and Exploration 
## Camden Bibro
# Date = 2024-11-03

## Load Libraries
import os
import mat73 # for loading pesky MATLAB structs!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat as lm
import pickle


# Set data folder paths
ctx_data_path = r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/full_ctx_data'
thalamus_data_path = r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/thalamus_data'

# Set save path 
save_path = r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/'

# Read in brain regions
regions = pd.read_csv(r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/region_names.csv')

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


def load_matlab_structs(data_path, data_array, target= "rest2_pcorr"):
    
    for i in range(0, len(data_path)):
        fMRI = []
        data = [] 
        try: 
            fMRI = mat73.loadmat(data_path[i]) # try to load with mat73 for MATLAB structs version 7.3
            # if this works then store the data in the array
            data = fMRI["fMRI_struct"]
            data_array[:,:,i] = data[target].astype(float)

        except TypeError: # if the file is not a MATLAB struct version 7.3
            fMRI = lm(data_path[i]) # Load with scipy.io instead
            data = fMRI["fMRI_struct"]
            data_array[:,:,i] = data[0,0][target].astype(float)
        except KeyError:
            print("warning! No data found for patient: ", i)
            continue
    return data_array


#%%
# LOAD IN BRAIN CORTICAL CONNECTIVITY DATA from MATLAB struct data 

# For CTX data (Cortex, Full-Brain)
file_names = np.array(os.listdir(ctx_data_path))
# Only use files with "_01" subscript * (01 indicates first (preoperative) scan)
file_names = np.array([name for name in file_names if "_01" in name])

pat_idx = np.array([index for index, name in enumerate(file_names) if "pat" in name])
con_idx = np.array([index for index, name in enumerate(file_names) if "con" in name])
pat_fpaths = np.array([ctx_data_path + "/" + str(path) + "/" + str(path) + "_parcellated.mat" for index, path in enumerate(file_names) if "pat" in path])
con_fpaths = np.array([ctx_data_path + "/" + str(path) + "/" + str(path) + "_parcellated.mat" for index, path in enumerate(file_names) if "con" in path])
pat_names = file_names[pat_idx]
con_names = file_names[con_idx]

# load in one file to get the dimensions of the data
tmp = lm(pat_fpaths[0])
tmp = tmp["fMRI_struct"]
tmp[0,0]["patID"]
num_nodes = tmp[0,0]["rest2_pcorr"].shape[0]

# Read the struct labels
# Initialize arrays to hold data
pats_ctx_data = np.zeros((num_nodes, num_nodes, len(pat_fpaths)))
cons_ctx_data = np.zeros((num_nodes, num_nodes, len(con_fpaths)))

# use the function to load in the data
pats_ctx_data = load_matlab_structs(pat_fpaths, pats_ctx_data)
cons_ctx_data = load_matlab_structs(con_fpaths, cons_ctx_data)

# Check missing data counts for each patient (3rd dimension)
pats_missing_data = np.isnan(pats_ctx_data).sum(axis=(0,1))
cons_missing_data = np.isnan(cons_ctx_data).sum(axis=(0,1))

# Find indicies where there is over 1700 missing data points
pat_missing_idx = np.where(cons_missing_data > 1800)
con_missing_idx = np.where(cons_missing_data > 1800)

# Remove patients with missing data
pats_ctx_data = np.delete(pats_ctx_data, pat_missing_idx, axis=2)
pat_names = np.delete(pat_names, pat_missing_idx, axis=0)
pat_fpaths = np.delete(pat_fpaths, pat_missing_idx)

cons_ctx_data = np.delete(cons_ctx_data, con_missing_idx, axis=2)
con_names = np.delete(con_names, con_missing_idx)
con_fpaths = np.delete(con_fpaths, con_missing_idx)


# Remove other unwanted regions (Keep only ctx and amygdala/himmpocampus)
# Add values 3,4,10,11 to the regions to keep (amygdala and hippocampus)
regions_to_keep = np.array([3,4,10,11])
regions_to_keep = np.append(regions_to_keep, np.arange(13, 81, 1))

cons_ctx_data = cons_ctx_data[regions_to_keep, :, :]
cons_ctx_data = cons_ctx_data[:, regions_to_keep, :]
pats_ctx_data = pats_ctx_data[regions_to_keep, :, :]
pats_ctx_data = pats_ctx_data[:, regions_to_keep, :]
ctx_regions = regions.iloc[regions_to_keep]

# Remove the diagonal from the data
for i in range(cons_ctx_data.shape[2]):
    np.fill_diagonal(cons_ctx_data[:, :, i], 0) 
for i in range(pats_ctx_data.shape[2]):
    np.fill_diagonal(pats_ctx_data[:, :, i], 0)

# Replace all missing values with 0 --> this is a good assumption since the data is zscored 

cons_ctx_data[np.isnan(cons_ctx_data)] = 0
pats_ctx_data[np.isnan(pats_ctx_data)] = 0

# Visualize the data distribution (Looks to be noramlly distributed)
plt.hist(cons_ctx_data[:,:,:].flatten(), bins=100, alpha=0.5, label='Controls')

pat_names_ctx  = pat_names
con_names_ctx = con_names


#%%
# LOAD IN THALAMIC CONNECTIVITY DATA

file_names = np.array(os.listdir(thalamus_data_path))
# Only use files with "_01" subscript * (01 indicates first (preoperative) scan)
file_names = np.array([name for name in file_names if "_01" in name])

pat_idx = np.array([index for index, name in enumerate(file_names) if "pat" in name])
con_idx = np.array([index for index, name in enumerate(file_names) if "con" in name])
pat_fpaths = np.array([thalamus_data_path + "/" + str(path) + "/" + str(path) + "_parcellated.mat" for index, path in enumerate(file_names) if "pat" in path])
con_fpaths = np.array([thalamus_data_path + "/" + str(path) + "/" + str(path) + "_parcellated.mat" for index, path in enumerate(file_names) if "con" in path])
thal_pat_names = file_names[pat_idx]
thal_con_names = file_names[con_idx] 

# Find indx where thal_pat_names and pat_names match
pat_idx = np.array([index for index, name in enumerate(thal_pat_names) if name in pat_names_ctx])
con_idx = np.array([index for index, name in enumerate(thal_con_names) if name in con_names_ctx])

# Make sure the names are the same
thal_pat_names = thal_pat_names[pat_idx]
thal_con_names = thal_con_names[con_idx]
thal_pat_names == pat_names_ctx
thal_con_names == con_names_ctx

# Great, now select only these names from the file paths
pat_fpaths = pat_fpaths[pat_idx]
con_fpaths = con_fpaths[con_idx] 


# load in one file to get the dimensions of the data
tmp = mat73.loadmat(pat_fpaths[0])
tmp = tmp["fMRI_struct"]
tmp["patID"]
num_nodes = tmp["rest2_pcorr"].shape

# Initialize arrays to hold data
pats_thal_data = np.zeros((num_nodes[0], num_nodes[1], len(pat_fpaths)))
cons_thal_data = np.zeros((num_nodes[0], num_nodes[1], len(con_fpaths)))

# use the function to load in the data
pats_thal_data = load_matlab_structs(pat_fpaths, pats_thal_data)
cons_thal_data = load_matlab_structs(con_fpaths, cons_thal_data)

# Replace all missing values with 0 --> this is a good assumption since the data is zscored
cons_thal_data[np.isnan(cons_thal_data)] = 0
pats_thal_data[np.isnan(pats_thal_data)] = 0

# View the data distribution
plt.hist(cons_thal_data[:,:,:].flatten(), bins=100, alpha=0.5, label='Controls')


# %%



patient_data_dict = {}
control_data_dict = {}

# Populate the dictionary with additional attributes
for i, pat_id in enumerate(pat_names):
    patient_data_dict[pat_id] = {
        "ctx_rest2_pcorr": pats_ctx_data[:, :, i],   # 72x72 matrix for that patient
        "thal_rest2_pcorr": pats_thal_data[:,:,i]    # 14x72 matrix for that patient
    }
    
for i, pat_id in enumerate(con_names):
    control_data_dict[pat_id] = {
        "ctx_rest2_pcorr": cons_ctx_data[:, :, i],   # 72x72 matrix for that control
        "thal_rest2_pcorr": cons_thal_data[:,:,i]    # 14x72 matrix for that control
    }
    
# save the dictionaries to a file 
with open(save_path + 'patient_data_dict.pkl', 'wb') as f:
    pickle.dump(patient_data_dict, f)
with open(save_path + 'control_data_dict.pkl', 'wb') as f:
    pickle.dump(control_data_dict, f)



# ! mkdir /Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/py_data_without_pickle
new_save_path = r'/Users/camdenbibro/Documents/VS_CODE/Thalamus_Network_Project/py_data_without_pickle/'

# Save the data as numpy arrays
np.save(new_save_path + 'pats_ctx_data.npy', pats_ctx_data)
np.save(new_save_path + 'cons_ctx_data.npy', cons_ctx_data)
np.save(new_save_path + 'pats_thal_data.npy', pats_thal_data)
np.save(new_save_path + 'cons_thal_data.npy', cons_thal_data)
np.save(new_save_path + 'ctx_regions.npy', ctx_regions)
np.save(new_save_path + 'pat_names.npy', pat_names)
np.save(new_save_path + 'con_names.npy', con_names)
# %%
