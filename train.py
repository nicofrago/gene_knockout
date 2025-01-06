import os
import glob
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from utils import plot_training
from torch.utils.data import TensorDataset, DataLoader
def train_model(
        model: nn.Sequential,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor, 
        y_val: torch.Tensor, 
        optimizer = optim.Adam, 
        loss_fn = nn.CrossEntropyLoss(),
        num_epochs = 50, 
        batch = 64,
        weights_dir = 'weights/simple_model'
):    
    os.makedirs(weights_dir, exist_ok=True)
    # model version
    v = len(glob.glob(f'{weights_dir}/*.pth'))
    v = str(v)
    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    # Validation dataset
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Ensure model is in training mode
    model.train()
    valloss = []
    trainloss = []
    for epoch in range(num_epochs):
        train_losses = []
        for batch_X, batch_y in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = loss_fn(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
        trainloss.append(sum(train_losses)/len(train_losses))
        # Validation phase (optional)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss = loss_fn(val_outputs, val_y)
                val_losses.append(val_loss.item())
            valloss.append(sum(val_losses)/len(val_losses))
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}, Validation Loss: {sum(val_losses)/len(val_losses)}")
        
        # Set back to training mode for next epoch
        model.train()

    save_model_in = f'{weights_dir}/modelv{v}.pth'
    torch.save(model.state_dict(), save_model_in)
    results_dict = {
        'trainloss': trainloss,
        'valloss': valloss  
    }
    model.eval()   
    pd.DataFrame(results_dict).to_csv(f'{weights_dir}/results_v{v}.csv', index=False)
    save_plot_in = f'{weights_dir}/loss_v{v}.png'
    plot_training(trainloss, valloss, num_epochs, save_plot_in)
    return model