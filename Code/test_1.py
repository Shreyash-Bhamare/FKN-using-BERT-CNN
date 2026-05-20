import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, BertModel
from torchvision import models, transforms
from collections import Counter
from tqdm import tqdm
import pandas as pd
from PIL import Image
# -------------------------
# Import FakeNewsDataset class
# -------------------------
from torch.serialization import add_safe_globals
from my_dataset_script import FakeNewsDataset 

# -------------------------
# Define the Hybrid Model

# -------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_fc = nn.Linear(self.bert.config.hidden_size, 256)

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask, image):
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_feat = self.bert_fc(text_feat)
        image_feat = self.cnn(image)

        combined = torch.cat((text_feat, image_feat), dim=1)
        x = self.dropout(combined)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------
# Load Saved Dataset (.pt)
# -------------------------
print("[INFO] Loading saved dataset (.pt)...")
dataset_path = r"C:/Users/Shreyash Bhamare/Desktop/IIIT Pune/Projects/Fake News Detection/data/fake_news_dataset.pt"  # Update path as needed
add_safe_globals([FakeNewsDataset])
dataset = torch.load("C:/Users/Shreyash Bhamare/Desktop/IIIT Pune/Projects/Fake News Detection/data/fake_news_dataset.pt", weights_only=False)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

# -------------------------
# Compute Class Weights
# -------------------------
labels = [sample["label"].item() for sample in dataset]
class_counts = Counter(labels)
class_weights = torch.tensor([
    1.0 / class_counts[0],
    1.0 / class_counts[1]
], dtype=torch.float)

# -------------------------
# Training Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = HybridModel().to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = AdamW(model.parameters(), lr=2e-5)

# -------------------------
# Training Loop
# -------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, image)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[INFO] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# -------------------------
# Save Trained Model
# -------------------------
torch.save(model.state_dict(), "hybrid_model.pt")
print("[INFO] Model saved as hybrid_model.pt")
