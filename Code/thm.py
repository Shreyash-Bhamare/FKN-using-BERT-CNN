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
# Define FakeNewsDataset class
# -------------------------
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_root_dir, max_length=128):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_root = image_root_dir
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = str(row["title"])
        desc = str(row["description"])
        combined_text = title + " [SEP] " + desc

        encoding = self.tokenizer(
            combined_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        img_path = os.path.join(self.image_root, row["media_source"], row["image_fn"])
        try:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.image_transform(image)
        except:
            image = torch.zeros((3, 224, 224))

        label = torch.tensor(row["is_fake"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label
        }

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
# Training function
# -------------------------~~
def train():
    csv_path = "C:/Users/Shreyash Bhamare/Desktop/IIIT Pune/Projects/Fake News Detection/data/evons_cleaned_filtered.csv"  # Adjust filename as needed
    image_root = "data/images/images"

    print("[INFO] Loading dataset from CSV...")
    dataset = FakeNewsDataset(csv_path, image_root)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    labels = [sample["label"].item() for sample in dataset]
    class_counts = Counter(labels)
    class_weights = torch.tensor([
        1.0 / class_counts[0],
        1.0 / class_counts[1]
    ], dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = HybridModel()
    model=model.cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=2e-5)

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

    torch.save(model.state_dict(), "hybrid_model.pt")
    print("[INFO] Model saved as hybrid_model.pt")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()
