
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from classifier import ASLClassifier

class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(data_dir, max_len=100):
    """
    Loads data from the data directory.
    Expected structure:
    data/
        class_1/
            0.npy
            1.npy
        class_2/
            ...
    """
    X = []
    y = []
    # Sort classes alphabetically
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_map = {label: idx for idx, label in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {classes}")
    
    total_samples = 0
    kept_samples = 0
    
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        for file in os.listdir(class_dir):
            if file.endswith('.npy'):
                sequence = np.load(os.path.join(class_dir, file))
                total_samples += 1
                
                # Filter zero frames
                # Check if row is all zeros
                non_zero_indices = ~np.all(sequence == 0, axis=1)
                sequence = sequence[non_zero_indices]
                
                if sequence.shape[0] == 0:
                    continue # Skip empty sequences
                
                kept_samples += 1
                
                # Pad or Truncate
                if sequence.shape[0] > max_len:
                    sequence = sequence[:max_len, :]
                elif sequence.shape[0] < max_len:
                    pad_len = max_len - sequence.shape[0]
                    padding = np.zeros((pad_len, sequence.shape[1]))
                    sequence = np.vstack((sequence, padding))
                
                X.append(sequence)
                y.append(class_map[label])
                
    print(f"Data Loading: Kept {kept_samples}/{total_samples} samples after filtering empty frames.")
    return np.array(X), np.array(y), classes

def train_model(data_dir='../data', model_path='../model.pth', epochs=50):
    X, y, classes = load_data(data_dir)
    print(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Number of classes: {len(classes)}")
    
    if len(classes) == 0:
        print("No data found! Please run collect_data.py first.")
        return

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = ASLDataset(X_train, y_train)
    test_dataset = ASLDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Init Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLClassifier(input_size=42, hidden_size=128, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {100 * correct / total:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
    # Save Model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    base_data_dir = os.path.join(os.path.dirname(__file__), '../data/processed_ipn')
    model_save_path = os.path.join(os.path.dirname(__file__), '../asl_model.pth')
    
    train_model(base_data_dir, model_save_path, args.epochs)
