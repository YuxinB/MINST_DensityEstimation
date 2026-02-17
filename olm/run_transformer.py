import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils as utils
import math
from tqdm import tqdm
import argparse
from rich.progress import Progress
import time

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
SEQ_LEN = 784
TRAIN_SIZE = 5000
TEST_SIZE = 200
SEED = 42

# architecture
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 1024
DROPOUT = 0.1

# vocabulary
VOCAB_SIZE = 3 # 0, 1, 2 (SOS)
SOS_TOKEN = 2

# training
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 200 

# -----------------------------------------------------------------------------
# Set seed
# -----------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_embedding', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()    
        
        # Embeddings
        x = self.token_embedding(x.long()) + self.position_embedding[:, :seq_len]
        x = self.dropout(x)
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer
        x = self.transformer_encoder(x, mask=mask, is_causal=True)
        
        # Output logits
        logits = self.fc_out(x)
        return logits

# -----------------------------------------------------------------------------
# Helper functions for train/test
# -----------------------------------------------------------------------------

def train_one_step(data, model, optimizer, criterion):
    data = data.to(DEVICE) # (B, 784), values 0 or 1
    batch_size = data.size(0)
    
    # Create SOS token
    sos = torch.full((batch_size, 1), SOS_TOKEN, device=DEVICE)
    
    # Input: [SOS, x_0, ..., x_782] (Length 784)
    # Target: [x_0, x_1, ..., x_783] (Length 784)
    
    input_seq = torch.cat([sos, data[:, :-1]], dim=1) # [SOS, x_0, ..., x_782]
    target_seq = data # [x_0, ..., x_783]
    
    optimizer.zero_grad()
    logits = model(input_seq) # (B, 784, Vocab)
    
    # Loss
    # Reshape logits to (B*784, Vocab) and target to (B*784)
    loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.long().reshape(-1))
    
    loss.backward()
    optimizer.step()

    return loss.item()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(loader):
        loss = train_one_step(data, model, optimizer, criterion)
        total_loss += loss    
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss:.4f}")
            
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_nll = 0
    total_pixels = 0
    
    for data, _ in loader:
        data = data.to(DEVICE)
        batch_size = data.size(0)
        
        sos = torch.full((batch_size, 1), SOS_TOKEN, device=DEVICE)
        input_seq = torch.cat([sos, data[:, :-1]], dim=1)
        target_seq = data
        
        logits = model(input_seq)
        logits[:, :, SOS_TOKEN] = -float('inf') # set logit for SOS to -inf
        
        # Compute Cross Entropy (NLL per token)
        nll_sum = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target_seq.long().reshape(-1), reduction='sum')
        total_nll += nll_sum.item()
        total_pixels += data.numel()
            
    # Average NLL per image
    nats_per_image = total_nll / len(loader.dataset)
    
    return nats_per_image

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['deterministic', 'stochastic'], default='stochastic')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', type=str, choices=['online', 'batched'], default='batched')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    dataset = torch.load(f"{args.dataset}_dataset.pt")
    train_data, test_data = dataset['train_data'], dataset['test_data']
    train_labels, test_labels = dataset['train_labels'], dataset['test_labels']
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=BATCH_SIZE, shuffle=False)

    model = Transformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS, 
        ff_dim=FF_DIM, 
        max_seq_len=SEQ_LEN + 1, 
        dropout=DROPOUT
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    print("Starting training...")
    if args.mode == 'online':
        nats_list = []
        with Progress() as progress:
            task = progress.add_task(f"[bold blue]{args.mode.capitalize()} Mode", total=len(train_data))
            start_time = time.perf_counter()
            for i, data in enumerate(train_data):
                train_loss = train_one_step(data.view(1, -1), model, optimizer, criterion)
                nats = evaluate(model, test_loader, criterion)
                nats_list.append(nats)
                progress.update(task, advance=1, description=f"[bold blue]{args.mode.capitalize()} Mode | Step {i+1} | Loss: {train_loss:.4f} | Nats: {nats:.4f}")
            end_time = time.perf_counter()
            training_time = end_time - start_time
            print(f"Training time: {training_time:.4f} seconds")
    elif args.mode == 'batched':
        nats_list = []
        with Progress() as progress:
            task = progress.add_task(f"[bold blue]{args.mode.capitalize()} Mode", total=EPOCHS)
            start_time = time.perf_counter()
            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                nats = evaluate(model, test_loader, criterion)
                nats_list.append(nats)
                progress.update(task, advance=1, description=f"[bold blue]{args.mode.capitalize()} Mode | Epoch {epoch} | Loss: {train_loss:.4f} | Nats: {nats:.4f}")
            end_time = time.perf_counter()
            training_time = end_time - start_time
            print(f"Training time: {training_time:.4f} seconds")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    result = {
        "nats_list": nats_list,
        "training_time": training_time,
        "dataset": args.dataset,
        "mode": args.mode,
        "total_params": total_params,
        "model_params": {
            "vocab_size": VOCAB_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM,
            "max_seq_len": SEQ_LEN + 1,
            "dropout": DROPOUT
        }
    }
    np.save(f"{args.dataset}_{args.mode}_transformer_results.npy", result)

    