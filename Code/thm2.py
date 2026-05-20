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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import numpy as np

# -------------------------
# Define FakeNewsDataset class
# -------------------------
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path_or_df, image_root_dir, max_length=128, augment=False):
        if isinstance(csv_path_or_df, str):
            self.df = pd.read_csv(csv_path_or_df).reset_index(drop=True)
        else:
            self.df = csv_path_or_df.reset_index(drop=True)

        self.image_root = image_root_dir
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Choose image transform
        if augment:
            self.image_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
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
    full_df = pd.read_csv(csv_path)

    train_df, temp_df = train_test_split(full_df, test_size=0.3, stratify=full_df["is_fake"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["is_fake"], random_state=42)

    train_dataset = FakeNewsDataset(train_df, image_root, augment=True)
    val_dataset = FakeNewsDataset(val_df, image_root)
    test_dataset = FakeNewsDataset(test_df, image_root)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    labels = [sample["label"].item() for sample in train_dataset]
    class_counts = Counter(labels)
    class_weights = torch.tensor([
        1.0 / class_counts[0],
        1.0 / class_counts[1]
    ], dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = HybridModel().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

    epochs = 5
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                image = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask, image)
                loss = loss_fn(outputs, labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[INFO] Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[INFO] Early stopping triggered.")
                break

    torch.save(model.state_dict(), "hybrid_model_2.pt", _use_new_zipfile_serialization=True)
    print("[INFO] Regularized model saved.")


    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate on test set
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[INFO] Evaluating on Test Set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, image)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"], digits=4)
    print("\n📊 Final Test Set Evaluation:")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    result_df = test_df.copy()
    result_df["predicted"] = all_preds
    result_df.to_csv("test_predictions_regularized.csv", index=False)
    print("[INFO] Regularized predictions saved to test_predictions_regularized.csv")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()
