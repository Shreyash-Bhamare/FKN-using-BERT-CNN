import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import os

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
            image = torch.zeros((3, 224, 224))  # fallback (should rarely happen now)

        # --- Label ---
        label = torch.tensor(row["is_fake"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label
        }
torch.serialization.add_safe_globals([FakeNewsDataset])