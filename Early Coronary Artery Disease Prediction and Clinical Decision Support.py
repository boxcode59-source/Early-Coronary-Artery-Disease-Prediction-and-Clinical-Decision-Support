# ==========================================================
# PEFMF - PART 1
# Data Collection + Edge-Based Privacy-Aware Data Distribution
# ==========================================================

# Install Libraries (Run once in Colab)
!pip -q install openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

OUTPUT_FOLDER = "Part1_Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

DATASET_PATH = "/content/cardiovascular.csv"   # Change your dataset path

df = pd.read_csv(DATASET_PATH)

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)

print(df.head())

print("\nDataset Shape :", df.shape)

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

df = df.drop_duplicates()

# ==========================================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================================

encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col].astype(str))

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

df.fillna(df.median(numeric_only=True), inplace=True)

# ==========================================================
# TARGET COLUMN
# ==========================================================

TARGET_COLUMN = "Target"      # <-- Change if necessary

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# CREATE EDGE DATASET
# ==========================================================

train_df = X_train.copy()
train_df[TARGET_COLUMN] = y_train.values

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================================
# EDGE PARTITIONING
# ==========================================================

NUM_EDGE_SERVERS = 5

edge_nodes = np.array_split(train_df, NUM_EDGE_SERVERS)

# ==========================================================
# SAVE EACH EDGE NODE
# ==========================================================

print("\nSaving Edge Nodes...\n")

for i, edge in enumerate(edge_nodes):

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"Edge_Node_{i+1}.csv"
    )

    edge.to_csv(filename, index=False)

    print(f"Edge {i+1} -> {edge.shape}")

# ==========================================================
# SIMULATED EDGE PARAMETERS
# ==========================================================

np.random.seed(42)

latency = np.random.uniform(5,15,NUM_EDGE_SERVERS)

energy = np.random.uniform(10,30,NUM_EDGE_SERVERS)

communication = np.random.uniform(2,8,NUM_EDGE_SERVERS)

# ==========================================================
# COST FUNCTION
# ==========================================================

w1 = 0.4
w2 = 0.3
w3 = 0.3

cost = (
    w1*latency +
    w2*energy +
    w3*communication
)

# ==========================================================
# SUMMARY TABLE
# ==========================================================

summary = pd.DataFrame({

    "Edge Server":[f"Edge-{i+1}" for i in range(NUM_EDGE_SERVERS)],

    "Latency(ms)":latency,

    "Energy":energy,

    "Communication":communication,

    "Distribution Cost":cost

})

print("\n")
print(summary)

summary.to_excel(
    os.path.join(
        OUTPUT_FOLDER,
        "Edge_Server_Summary.xlsx"
    ),
    index=False
)

# ==========================================================
# BAR GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(summary["Edge Server"],summary["Distribution Cost"])

plt.title("Edge Distribution Cost")

plt.xlabel("Edge Server")

plt.ylabel("Cost")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Edge_Distribution_Cost.png"
    ),
    dpi=600
)

plt.show()

# ==========================================================
# DISTRIBUTED HEALTHCARE REPOSITORY
# ==========================================================

distributed_repository = {}

for i in range(NUM_EDGE_SERVERS):

    distributed_repository[f"Edge-{i+1}"] = edge_nodes[i]

print("\nDistributed Repository Created Successfully")

for key,value in distributed_repository.items():

    print(key,"->",value.shape)

print("\nPART-1 COMPLETED SUCCESSFULLY")
# ==========================================================
# PEFMF - PART 2
# TinyFedX Secure Federated Learning
# ==========================================================

# Install libraries (Run once)
!pip -q install xgboost

import os
import base64
import pickle
import hashlib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# ==========================================================
# SETTINGS
# ==========================================================

EDGE_FOLDER = "Part1_Output"      # Output folder created in Part-1
TARGET_COLUMN = "Target"

NUM_EDGES = 5

# ==========================================================
# LOAD EDGE DATASETS
# ==========================================================

edge_datasets = []

for i in range(NUM_EDGES):

    file = os.path.join(EDGE_FOLDER, f"Edge_Node_{i+1}.csv")

    df = pd.read_csv(file)

    edge_datasets.append(df)

print("Loaded", len(edge_datasets), "Edge Datasets")

# ==========================================================
# INITIAL GLOBAL MODEL
# ==========================================================

global_model = SGDClassifier(
    loss="log_loss",
    random_state=42
)

classes = np.array([0,1])

# ==========================================================
# LOCAL TRAINING
# ==========================================================

encrypted_updates = []

local_accuracies = []

print("\n==============================")
print("Local Model Training")
print("==============================")

for i, df in enumerate(edge_datasets):

    X = df.drop(TARGET_COLUMN, axis=1).values
    y = df[TARGET_COLUMN].values

    model = SGDClassifier(
        loss="log_loss",
        random_state=42
    )

    model.partial_fit(X, y, classes=classes)

    prediction = model.predict(X)

    acc = accuracy_score(y, prediction)

    local_accuracies.append(acc)

    print(f"Edge-{i+1} Accuracy : {acc:.4f}")

    # ===========================================
    # Tiny Encryption (Simulation)
    # ===========================================

    parameters = {

        "coef": model.coef_,

        "bias": model.intercept_

    }

    serialized = pickle.dumps(parameters)

    encrypted = base64.b64encode(serialized)

    encrypted_updates.append(encrypted)

# ==========================================================
# SERVER SIDE AGGREGATION
# ==========================================================

print("\n==============================")
print("Federated Aggregation")
print("==============================")

coef_list = []

bias_list = []

for update in encrypted_updates:

    decoded = base64.b64decode(update)

    params = pickle.loads(decoded)

    coef_list.append(params["coef"])

    bias_list.append(params["bias"])

global_coef = np.mean(coef_list, axis=0)

global_bias = np.mean(bias_list, axis=0)

global_model.coef_ = global_coef
global_model.intercept_ = global_bias
global_model.classes_ = classes

print("Global Model Updated Successfully")

# ==========================================================
# COMMUNICATION COST
# ==========================================================

communication_cost = 0

for update in encrypted_updates:

    communication_cost += len(update)

print("\nCommunication Cost (Bytes):", communication_cost)

# ==========================================================
# BLOCK HASH (Integrity Simulation)
# ==========================================================

hash_value = hashlib.sha256(
    pickle.dumps({
        "coef":global_coef,
        "bias":global_bias
    })
).hexdigest()

print("\nGlobal Model Hash")
print(hash_value)

# ==========================================================
# SAVE GLOBAL MODEL
# ==========================================================

os.makedirs("Part2_Output", exist_ok=True)

with open("Part2_Output/global_model.pkl","wb") as f:

    pickle.dump(global_model,f)

# ==========================================================
# SAVE TRAINING SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Edge":[f"Edge-{i+1}" for i in range(NUM_EDGES)],

    "Local Accuracy":local_accuracies

})

summary.loc[len(summary)] = [

    "Global",

    np.mean(local_accuracies)

]

summary.to_excel(

    "Part2_Output/Federated_Training_Summary.xlsx",

    index=False

)

print("\nAverage Federated Accuracy :",

      round(np.mean(local_accuracies)*100,2),

      "%")

print("\nGlobal Model Saved Successfully")

print("\nPART-2 COMPLETED SUCCESSFULLY")

# ==========================================================
# PEFMF - PART 3
# Differential Privacy 2.0 + Lightweight Blockchain Authentication
# ==========================================================

import os
import pickle
import hashlib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

OUTPUT_FOLDER = "Part3_Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD GLOBAL MODEL FROM PART-2
# ==========================================================

MODEL_PATH = "Part2_Output/global_model.pkl"

with open(MODEL_PATH, "rb") as f:
    global_model = pickle.load(f)

print("="*60)
print("Global Federated Model Loaded Successfully")
print("="*60)

# ==========================================================
# DIFFERENTIAL PRIVACY PARAMETERS
# ==========================================================

epsilon = 1.0        # Privacy Budget
delta = 1e-5
sensitivity = 1.0

noise_scale = sensitivity / epsilon

print("\nPrivacy Parameters")
print("---------------------")
print("Epsilon :", epsilon)
print("Delta   :", delta)
print("Noise Scale :", noise_scale)

# ==========================================================
# ORIGINAL PARAMETERS
# ==========================================================

original_coef = global_model.coef_.copy()
original_bias = global_model.intercept_.copy()

# ==========================================================
# GAUSSIAN NOISE
# ==========================================================

coef_noise = np.random.normal(
    0,
    noise_scale,
    original_coef.shape
)

bias_noise = np.random.normal(
    0,
    noise_scale,
    original_bias.shape
)

private_coef = original_coef + coef_noise
private_bias = original_bias + bias_noise

# ==========================================================
# UPDATE MODEL
# ==========================================================

global_model.coef_ = private_coef
global_model.intercept_ = private_bias

print("\nDifferential Privacy Applied Successfully")

# ==========================================================
# SAVE PRIVACY MODEL
# ==========================================================

privacy_model_path = os.path.join(
    OUTPUT_FOLDER,
    "private_global_model.pkl"
)

with open(privacy_model_path, "wb") as f:
    pickle.dump(global_model, f)

# ==========================================================
# LIGHTWEIGHT BLOCKCHAIN
# ==========================================================

class Block:

    def __init__(self, index, timestamp, data, previous_hash):

        self.index = index

        self.timestamp = timestamp

        self.data = data

        self.previous_hash = previous_hash

        self.hash = self.calculate_hash()

    def calculate_hash(self):

        block = str(
            self.index
        ) + str(
            self.timestamp
        ) + str(
            self.data
        ) + str(
            self.previous_hash
        )

        return hashlib.sha256(
            block.encode()
        ).hexdigest()


# ==========================================================
# GENESIS BLOCK
# ==========================================================

blockchain = []

genesis = Block(

    0,

    time.time(),

    "Genesis Block",

    "0"

)

blockchain.append(genesis)

# ==========================================================
# ADD MODEL UPDATE BLOCK
# ==========================================================

model_information = {

    "coef_shape": private_coef.shape,

    "bias_shape": private_bias.shape,

    "privacy_budget": epsilon

}

block = Block(

    1,

    time.time(),

    model_information,

    genesis.hash

)

blockchain.append(block)

# ==========================================================
# VERIFY BLOCKCHAIN
# ==========================================================

valid = True

for i in range(1, len(blockchain)):

    if blockchain[i].previous_hash != blockchain[i-1].hash:

        valid = False

print("\nBlockchain Status :", valid)

# ==========================================================
# SAVE BLOCKCHAIN
# ==========================================================

block_data = []

for blk in blockchain:

    block_data.append({

        "Index": blk.index,

        "Timestamp": blk.timestamp,

        "Previous Hash": blk.previous_hash,

        "Current Hash": blk.hash

    })

block_df = pd.DataFrame(block_data)

block_df.to_excel(

    os.path.join(

        OUTPUT_FOLDER,

        "Blockchain.xlsx"

    ),

    index=False

)

# ==========================================================
# PRIVACY SUMMARY
# ==========================================================

privacy_summary = pd.DataFrame({

    "Parameter":[

        "Epsilon",

        "Delta",

        "Noise Scale",

        "Blockchain Valid"

    ],

    "Value":[

        epsilon,

        delta,

        noise_scale,

        valid

    ]

})

privacy_summary.to_excel(

    os.path.join(

        OUTPUT_FOLDER,

        "Privacy_Summary.xlsx"

    ),

    index=False

)

# ==========================================================
# PLOT NOISE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(7,5))

plt.hist(

    coef_noise.flatten(),

    bins=30

)

plt.title("Gaussian Noise Distribution")

plt.xlabel("Noise")

plt.ylabel("Frequency")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "Differential_Privacy_Noise.png"

    ),

    dpi=600

)

plt.show()

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\n")
print("="*60)
print("Differential Privacy Completed")
print("Blockchain Authentication Completed")
print("Private Model Saved")
print("Output Folder :", OUTPUT_FOLDER)
print("="*60)

# ==========================================================
# PEFMF - PART 4
# Lightweight Clinical Data Preprocessing
# Tiny Isolation Filtering + Missing Value Reconstruction
# + Quantized Normalization
# ==========================================================

# Install Required Libraries
!pip -q install openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

OUTPUT_FOLDER = "Part4_Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD EDGE DATASET
# ==========================================================

DATASET = "Part1_Output/Edge_Node_1.csv"     # Change if required

df = pd.read_csv(DATASET)

print("="*60)
print("Original Dataset Shape :", df.shape)
print("="*60)

# ==========================================================
# SAVE ORIGINAL COPY
# ==========================================================

original_rows = len(df)

# ==========================================================
# SEPARATE TARGET COLUMN
# ==========================================================

TARGET = "Target"

X = df.drop(columns=[TARGET])

y = df[TARGET]

# ==========================================================
# STEP 1 : Tiny Isolation Filtering
# ==========================================================

print("\nApplying Tiny Isolation Filtering...")

iso = IsolationForest(

    contamination=0.03,

    random_state=42

)

prediction = iso.fit_predict(X)

mask = prediction == 1

X = X[mask]

y = y[mask]

filtered_rows = len(X)

print("Remaining Samples :", filtered_rows)

# ==========================================================
# STEP 2 : Artificial Missing Values
# (Simulation)
# ==========================================================

print("\nGenerating Missing Values (Simulation)...")

np.random.seed(42)

X_missing = X.copy()

for column in X_missing.columns:

    index = X_missing.sample(frac=0.05).index

    X_missing.loc[index, column] = np.nan

print("Missing Values Generated")

# ==========================================================
# STEP 3 : TinyTab Transformer Imputation
# (Lightweight KNN Approximation)
# ==========================================================

print("\nReconstructing Missing Values...")

imputer = KNNImputer(

    n_neighbors=3

)

X_imputed = pd.DataFrame(

    imputer.fit_transform(X_missing),

    columns=X_missing.columns

)

print("Missing Values Reconstructed")

# ==========================================================
# STEP 4 : Quantized Normalization
# ==========================================================

print("\nApplying Quantized Normalization...")

scaler = MinMaxScaler()

normalized = scaler.fit_transform(X_imputed)

BITS = 8

levels = (2**BITS)-1

quantized = np.round(normalized*levels)/levels

X_final = pd.DataFrame(

    quantized,

    columns=X_imputed.columns

)

# ==========================================================
# COMBINE FEATURES + TARGET
# ==========================================================

processed = X_final.copy()

processed[TARGET] = y.values

print("\nProcessed Dataset Shape :", processed.shape)

# ==========================================================
# SAVE DATASET
# ==========================================================

processed.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "Preprocessed_Dataset.csv"

    ),

    index=False

)

# ==========================================================
# PREPROCESSING SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric":[

        "Original Samples",

        "Samples After Filtering",

        "Noise Removed",

        "Missing Values",

        "Quantization Bits"

    ],

    "Value":[

        original_rows,

        filtered_rows,

        original_rows-filtered_rows,

        X_missing.isnull().sum().sum(),

        BITS

    ]

})

summary.to_excel(

    os.path.join(

        OUTPUT_FOLDER,

        "Preprocessing_Summary.xlsx"

    ),

    index=False

)

print(summary)

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(figsize=(7,5))

bars = [

    original_rows,

    filtered_rows

]

labels = [

    "Original",

    "Filtered"

]

plt.bar(labels,bars)

plt.title("Tiny Isolation Filtering")

plt.ylabel("Number of Samples")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "Filtering_Result.png"

    ),

    dpi=600

)

plt.show()

# ==========================================================
# FEATURE DISTRIBUTION
# ==========================================================

feature = processed.columns[0]

plt.figure(figsize=(7,5))

plt.hist(processed[feature],bins=25)

plt.title(f"Normalized Feature Distribution : {feature}")

plt.xlabel("Normalized Value")

plt.ylabel("Frequency")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "Feature_Distribution.png"

    ),

    dpi=600

)

plt.show()

# ==========================================================
# COMPLETED
# ==========================================================

print("\n"+"="*60)
print("PART 4 COMPLETED SUCCESSFULLY")
print("Output Folder :", OUTPUT_FOLDER)
print("="*60)

# ==========================================================
# PEFMF - PART 5
# TinyDiffusion-CAD Data Augmentation
# ==========================================================

# Install Library
!pip -q install openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

OUTPUT_FOLDER = "Part5_Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD PREPROCESSED DATASET
# ==========================================================

DATASET = "Part4_Output/Preprocessed_Dataset.csv"

df = pd.read_csv(DATASET)

print("="*60)
print("Dataset Loaded Successfully")
print("Original Shape :", df.shape)
print("="*60)

TARGET = "Target"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# ==========================================================
# TINYDIFFUSION PARAMETERS
# ==========================================================

NUM_SYNTHETIC = len(X)

NOISE_LEVEL = 0.05

TIMESTEPS = 10

np.random.seed(42)

# ==========================================================
# FORWARD DIFFUSION
# ==========================================================

print("\nGenerating Noisy Samples...")

synthetic_samples = []

for i in range(NUM_SYNTHETIC):

    sample = X.iloc[i].values.astype(float)

    noisy = sample.copy()

    for t in range(TIMESTEPS):

        noise = np.random.normal(
            0,
            NOISE_LEVEL,
            size=noisy.shape
        )

        noisy = noisy + noise

    synthetic_samples.append(noisy)

synthetic_samples = np.array(synthetic_samples)

print("Forward Diffusion Completed")

# ==========================================================
# REVERSE DIFFUSION (LIGHTWEIGHT)
# ==========================================================

print("\nApplying Reverse Diffusion...")

reverse_samples = synthetic_samples.copy()

for t in range(TIMESTEPS):

    reverse_samples = (
        reverse_samples * 0.95 +
        X.values * 0.05
    )

print("Reverse Diffusion Completed")

# ==========================================================
# CLIP VALUES
# ==========================================================

reverse_samples = np.clip(reverse_samples,0,1)

# ==========================================================
# CREATE SYNTHETIC DATAFRAME
# ==========================================================

synthetic_df = pd.DataFrame(
    reverse_samples,
    columns=X.columns
)

synthetic_df[TARGET] = y.values

# ==========================================================
# COMBINE DATASETS
# ==========================================================

augmented_df = pd.concat(
    [df, synthetic_df],
    ignore_index=True
)

print("\nAugmented Shape :", augmented_df.shape)

# ==========================================================
# SAVE DATASETS
# ==========================================================

synthetic_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Synthetic_Data.csv"
    ),
    index=False
)

augmented_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "Augmented_Dataset.csv"
    ),
    index=False
)

# ==========================================================
# AUGMENTATION SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric":[

        "Original Samples",

        "Synthetic Samples",

        "Final Samples",

        "Noise Level",

        "Diffusion Steps"

    ],

    "Value":[

        len(df),

        len(synthetic_df),

        len(augmented_df),

        NOISE_LEVEL,

        TIMESTEPS

    ]

})

summary.to_excel(
    os.path.join(
        OUTPUT_FOLDER,
        "TinyDiffusion_Summary.xlsx"
    ),
    index=False
)

print(summary)

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(figsize=(7,5))

plt.bar(
    ["Original","Synthetic","Final"],
    [
        len(df),
        len(synthetic_df),
        len(augmented_df)
    ]
)

plt.title("TinyDiffusion-CAD Augmentation")

plt.ylabel("Samples")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Augmentation_Result.png"
    ),
    dpi=600
)

plt.show()

# ==========================================================
# FEATURE COMPARISON
# ==========================================================

feature = X.columns[0]

plt.figure(figsize=(8,5))

plt.hist(
    X[feature],
    bins=25,
    alpha=0.6,
    label="Original"
)

plt.hist(
    synthetic_df[feature],
    bins=25,
    alpha=0.6,
    label="Synthetic"
)

plt.legend()

plt.title("Feature Distribution After TinyDiffusion")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Feature_Comparison.png"
    ),
    dpi=600
)

plt.show()

# ==========================================================
# COMPLETED
# ==========================================================

print("\n"+"="*60)
print("TinyDiffusion-CAD Completed Successfully")
print("Output Folder :", OUTPUT_FOLDER)
print("="*60)


# ==========================================================
# PEFMF - PART 6A
# CardioLiteGraph Framework (CLGF)
# Graph Construction + Adjacency Matrix + Edge Index
# ==========================================================

# Install Libraries
!pip -q install networkx openpyxl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

OUTPUT_FOLDER = "Part6A_Output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# LOAD AUGMENTED DATASET
# ==========================================================

DATASET = "Part5_Output/Augmented_Dataset.csv"

df = pd.read_csv(DATASET)

print("="*60)
print("Dataset Loaded Successfully")
print("="*60)
print(df.shape)

# ==========================================================
# TARGET COLUMN
# ==========================================================

TARGET = "Target"

X = df.drop(columns=[TARGET])

y = df[TARGET]

print("\nFeature Matrix :", X.shape)

# ==========================================================
# NORMALIZE FEATURES
# ==========================================================

X = (X-X.min())/(X.max()-X.min()+1e-8)

# ==========================================================
# BUILD KNN GRAPH
# ==========================================================

K = 5

knn = NearestNeighbors(
    n_neighbors=K,
    metric="euclidean"
)

knn.fit(X)

distance, indices = knn.kneighbors(X)

print("\nGraph Construction Completed")

# ==========================================================
# CREATE ADJACENCY MATRIX
# ==========================================================

N = len(X)

adjacency = np.zeros((N,N))

for i in range(N):

    for j in indices[i]:

        adjacency[i,j]=1
        adjacency[j,i]=1

print("Adjacency Matrix Shape :",adjacency.shape)

# ==========================================================
# SAVE ADJACENCY MATRIX
# ==========================================================

adj_df = pd.DataFrame(adjacency)

adj_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "Adjacency_Matrix.csv"
    ),

    index=False
)

# ==========================================================
# EDGE INDEX
# ==========================================================

edge_index = np.array(np.where(adjacency==1))

print("Edge Index Shape :",edge_index.shape)

edge_df = pd.DataFrame({

    "Source":edge_index[0],

    "Destination":edge_index[1]

})

edge_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "Edge_Index.csv"
    ),

    index=False
)

# ==========================================================
# NODE FEATURE MATRIX
# ==========================================================

node_features = X.copy()

node_features.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "Node_Features.csv"
    ),

    index=False
)

print("Node Feature Shape :",node_features.shape)

# ==========================================================
# COSINE SIMILARITY MATRIX
# ==========================================================

similarity = cosine_similarity(node_features)

similarity_df = pd.DataFrame(similarity)

similarity_df.to_csv(

    os.path.join(
        OUTPUT_FOLDER,
        "Similarity_Matrix.csv"
    ),

    index=False
)

# ==========================================================
# NETWORKX GRAPH
# ==========================================================

G = nx.Graph()

for i in range(N):

    G.add_node(i,label=int(y.iloc[i]))

for i in range(edge_index.shape[1]):

    G.add_edge(

        int(edge_index[0,i]),

        int(edge_index[1,i])

    )

print("\nTotal Nodes :",G.number_of_nodes())

print("Total Edges :",G.number_of_edges())

# ==========================================================
# GRAPH INFORMATION
# ==========================================================

graph_info = pd.DataFrame({

    "Property":[

        "Nodes",

        "Edges",

        "Average Degree",

        "Graph Density"

    ],

    "Value":[

        G.number_of_nodes(),

        G.number_of_edges(),

        np.mean([d for n,d in G.degree()]),

        nx.density(G)

    ]

})

graph_info.to_excel(

    os.path.join(

        OUTPUT_FOLDER,

        "Graph_Information.xlsx"

    ),

    index=False

)

print(graph_info)

# ==========================================================
# VISUALIZE GRAPH
# ==========================================================

plt.figure(figsize=(10,8))

pos = nx.spring_layout(

    G,

    seed=42

)

nx.draw_networkx_nodes(

    G,

    pos,

    node_size=10,

    node_color=y,

    cmap="coolwarm"

)

nx.draw_networkx_edges(

    G,

    pos,

    alpha=0.25,

    width=0.3

)

plt.title("CardioLiteGraph Framework (CLGF)")

plt.axis("off")

plt.tight_layout()

plt.savefig(

    os.path.join(

        OUTPUT_FOLDER,

        "CardioLiteGraph.png"

    ),

    dpi=600

)

plt.show()

# ==========================================================
# SAVE GRAPH
# ==========================================================

nx.write_gpickle(

    G,

    os.path.join(

        OUTPUT_FOLDER,

        "CardioLiteGraph.gpickle"

    )

)

print("\n====================================================")

print("PART-6A COMPLETED SUCCESSFULLY")

print("Output Folder :",OUTPUT_FOLDER)

print("====================================================")

# =========================================================
# PART 6B - MODEL EVALUATION & EXPORT
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve
)

# =========================================================
# INPUT (FROM YOUR MODEL)
# y_test -> true labels
# y_pred -> predicted labels
# y_prob -> predicted probabilities (for ROC/PR)
# =========================================================

# Example placeholders (REMOVE if already defined)
# y_test = ...
# y_pred = ...
# y_prob = model.predict_proba(X_test)[:,1]

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================
output_dir = "results_part6B"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# METRICS
# =========================================================
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [acc, pre, rec, f1]
})

metrics_path = os.path.join(output_dir, "metrics.xlsx")
metrics_df.to_excel(metrics_path, index=False)

print("===== MODEL METRICS =====")
print(metrics_df)

# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(output_dir, "classification_report.xlsx")
report_df.to_excel(report_path)

# =========================================================
# ROC CURVE
# =========================================================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

roc_path = os.path.join(output_dir, "roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.show()

# =========================================================
# PRECISION - RECALL CURVE
# =========================================================
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

pr_path = os.path.join(output_dir, "pr_curve.png")
plt.savefig(pr_path, dpi=300)
plt.show()

# =========================================================
# SAVE ALL RESULTS INTO ONE ZIP FILE
# =========================================================
zip_path = "results_part6B.zip"

with zipfile.ZipFile(zip_path, 'w') as zipf:
    zipf.write(metrics_path)
    zipf.write(report_path)
    zipf.write(cm_path)
    zipf.write(roc_path)
    zipf.write(pr_path)

print("ZIP file created:", zip_path)