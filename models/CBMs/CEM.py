import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class CEM_embeddings(nn.Module):
    def __init__(self, input_dim, n_concepts=None, hidden_dim=64, embedding_dim=16, 
                 depth=2, output_dim=2, concept_names=None, dropout=0.2, feature_names=None):
        super().__init__()
        
        self.n_concepts = n_concepts
        if concept_names is not None:
            self.n_concepts = len(concept_names)
  
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim - self.n_concepts 
        
        positive_concept_networks = []
        negative_concept_networks = []

        for concept_i in range(self.n_concepts):    
            # Positive network
            pos_layers = []
            curr_in = self.input_dim
            for _ in range(depth):
                pos_layers.append(nn.Linear(curr_in, hidden_dim))
                pos_layers.append(nn.ReLU())
                pos_layers.append(nn.Dropout(dropout))
                curr_in = hidden_dim
            pos_layers.append(nn.Linear(curr_in, embedding_dim))
            positive_concept_networks.append(nn.Sequential(*pos_layers))
            
            # Negative network
            neg_layers = []
            curr_in = self.input_dim
            for _ in range(depth):
                neg_layers.append(nn.Linear(curr_in, hidden_dim))
                neg_layers.append(nn.ReLU())
                neg_layers.append(nn.Dropout(dropout))
                curr_in = hidden_dim
            neg_layers.append(nn.Linear(curr_in, embedding_dim))
            negative_concept_networks.append(nn.Sequential(*neg_layers))
            
        self.positive_concept_networks = nn.ModuleList(positive_concept_networks)
        self.negative_concept_networks = nn.ModuleList(negative_concept_networks)

    def forward(self, x):
        positive_concept_outputs = []
        negative_concept_outputs = []
        
        for i, (pos_net, neg_net) in enumerate(zip(self.positive_concept_networks, self.negative_concept_networks)):
            positive_concept_outputs.append(pos_net(x[:, :, i]))
            negative_concept_outputs.append(neg_net(x[:, :, i]))
            
        # Stack to keep concept dimension clear: [batch, embedding_dim, n_concepts]
        pos_out = torch.stack(positive_concept_outputs, dim=2)
        neg_out = torch.stack(negative_concept_outputs, dim=2)
        return pos_out, neg_out

class CEM_REGRESSION(nn.Module):
    def __init__(self, n_concepts, embedding_dim, hidden_dim=64, depth=2, dropout=0.2):
        super().__init__()
        self.n_concepts = n_concepts
        # Each concept network now receives (pos_embed + neg_embed)
        self.input_dim = embedding_dim * 2 

        concept_networks = []
        for concept_i in range(self.n_concepts):
            layers = []
            curr_in = self.input_dim
            for _ in range(depth):
                layers.append(nn.Linear(curr_in, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                curr_in = hidden_dim
            layers.append(nn.Linear(curr_in, 1))
            concept_networks.append(nn.Sequential(*layers))
            
        self.concept_networks = nn.ModuleList(concept_networks)

    def forward(self, x):
        concept_outputs = []
        for i, concept_net in enumerate(self.concept_networks):
            concept_outputs.append(concept_net(x[:, :, i]))
        return torch.cat(concept_outputs, dim=1)

class CEM(nn.Module):
    def __init__(self, input_dim, n_concepts=None, hidden_dim=64, embedding_dim=16, 
                 depth=2, output_dim=2, concept_names=None, dropout=0.2, feature_names=None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_concepts = n_concepts if concept_names is None else len(concept_names)

        self.cem_embeddings = CEM_embeddings(
            input_dim=input_dim, n_concepts=n_concepts, hidden_dim=hidden_dim,
            embedding_dim=embedding_dim, depth=depth, concept_names=concept_names, dropout=dropout
        )
        
        self.cem_regression = CEM_REGRESSION(
            n_concepts=self.n_concepts, embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim, depth=depth, dropout=dropout
        )

    def forward(self, x, interventions=None):
        # 1. Get pos/neg embeddings: [batch, embedding_dim, n_concepts]
        pos_embeds, neg_embeds = self.cem_embeddings(x)
        
        # 2. Prepare input for regression (concat pos and neg for each concept)
        # Shape becomes [batch, embedding_dim * 2, n_concepts]
        combined_for_reg = torch.cat([pos_embeds, neg_embeds], dim=1)
        
        # 3. Predict concepts
        predicted_concepts = self.cem_regression(combined_for_reg)
        
        # 4. Use predicted or provided intervention weights
        if interventions is None:    
            weights = torch.sigmoid(predicted_concepts).unsqueeze(1) # [batch, 1, n_concepts]
        else:
            weights = torch.sigmoid(interventions).unsqueeze(1)

        # 5. Weighted sum of embeddings
        # Result shape: [batch, embedding_dim, n_concepts]
        final_embeddings = (pos_embeds * weights) + (neg_embeds * (1 - weights))

        # Flatten embeddings for the next layer if needed: [batch, embedding_dim * n_concepts]
        final_embeddings_flat = final_embeddings.view(x.size(0), -1)

        return final_embeddings_flat, predicted_concepts

# --- Test Block ---
if __name__ == "__main__":
    batch_size = 4
    n_concepts = 3
    input_dim_param = 10  # Raw input features (including space for concepts)
    actual_data_dim = 7   # 10 - 3 = 7 (the features used to predict concepts)
    embedding_dim = 16

    # Input x: [batch, features_per_concept, n_concepts]
    x = torch.randn(batch_size, actual_data_dim, n_concepts)

    cem_model = CEM(input_dim=input_dim_param, n_concepts=n_concepts, embedding_dim=embedding_dim)

    embeddings, concepts = cem_model(x)

    print(f"Embeddings shape: {embeddings.shape}") # Expected: [4, 48] (3 concepts * 16 dim)
    print(f"Concepts shape:   {concepts.shape}")   # Expected: [4, 3]
    print("concepts :", concepts)