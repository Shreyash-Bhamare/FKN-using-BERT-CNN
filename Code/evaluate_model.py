import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from thm import HybridModel 

# -------------------------------
# Define the custom FakeNewsDataset class
# -------------------------------
class FakeNewsDataset(Dataset):
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

        # --- Text (BERT input) ---
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

        # --- Image (CNN input) ---
        img_path = os.path.join(self.image_root, row["media_source"], row["image_fn"])
        try:
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.image_transform(image)
        except:
            image = torch.zeros((3, 224, 224))  # fallback

        # --- Label ---
        label = torch.tensor(row["is_fake"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label
        }

# -------------------------------
# Step 1: Set device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Step 2: Load test dataset
# -------------------------------
test_dataset = FakeNewsDataset(
    csv_path="C:/Users/Shreyash Bhamare/Desktop/IIIT Pune/Projects/Fake News Detection/data/evons_test.csv",
    image_root_dir="data/images/images"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

# -------------------------------
# Step 3: Load trained model
# -------------------------------
model = HybridModel()
model.load_state_dict(torch.load("hybrid_model.pt", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Step 4: Evaluation function
# -------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Evaluating"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, image)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4)
    }

# -------------------------------
# Step 5: Run evaluation
# -------------------------------
if __name__ == "__main__":
    metrics = evaluate_model(model, test_loader, device)

    print("\n🔧 Summary of Metrics:")
    for k, v in metrics.items():
        print(f"{k.title()}: {v}")

