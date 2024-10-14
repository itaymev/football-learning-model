import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from classes.FootballPredictorClass import FootballPredictor
from classes.EarlyStoppingClass import EarlyStopping
from classes.PreprocessorClass import Preprocessor

FILE_PATH = 'football-learning-model/data/past-data.csv'

# Initialize the preprocessor and get preprocessed data
preprocessor = Preprocessor(FILE_PATH)
features, target = preprocessor.get_preprocessed_data()
preprocessor.check_for_nan()

# Cross-validation and learning rate experimentation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
learning_rates = [3e-06]
best_accuracy = 0
best_lr = None

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f"Fold {fold + 1}")
        
        # Split the data
        X_train_fold, X_val_fold = features[train_idx], features[val_idx]
        y_train_fold, y_val_fold = target[train_idx], target[val_idx]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize the model
        input_dim = X_train_fold.shape[1]
        model = FootballPredictor(input_dim)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        
        # Training loop with early stopping
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                if torch.isnan(loss):
                    raise ValueError("Loss is NaN. Please check the input data and model.")
                loss.backward()
                optimizer.step()
            
            # Evaluate the model
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1) 
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            val_loss /= len(val_loader)
            accuracy = correct / total

            # See README.md for definitions on all this (pushing this code)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Accuracy: {accuracy * 100:.2f}%')
            
            if early_stopping(val_loss):
                print("Early stopping triggered") # bad probably
                break
        
        # Print fold accuracy
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")

        torch.save(model.state_dict(), f'Project 1/folds/model_fold_{fold + 1}.pth') # This is saving to a local location I need to fix this, maybe not idk how much time I have
        print(f"Iteration {fold} - Fold {fold + 1} saved to: model_fold_{fold + 1}.pth")

    # Calculate average accuracy for the current learning rate
    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Average Cross-Validation Accuracy with learning rate {lr}: {average_accuracy * 100:.2f}%")
    
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_lr = lr

print(f"Best Learning Rate: {best_lr}, Best Accuracy: {best_accuracy * 100:.2f}%")