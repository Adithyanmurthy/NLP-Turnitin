"""
Person 3 - Data Loader
Loads preprocessed datasets for training.

Data sources (checked in order):
  1. person_3/data/train.jsonl  (Person 3's own downloader output)
  2. person_1/data/splits/       (Person 1's preprocessed paraphrase data — fallback)
"""
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import DATA_DIR, TRAINING_CONFIG

# Person 1's splits directory (fallback data source)
PERSON1_SPLITS_DIR = Path(__file__).parent.parent / "person_1" / "data" / "splits"

# Paraphrase datasets that Person 1 preprocesses and Person 3 can use
PERSON1_PARAPHRASE_DATASETS = ["paws", "qqp", "mrpc", "paranmt", "wikisplit"]

class ParaphraseDataset(Dataset):
    """Dataset for paraphrase training"""
    
    def __init__(self, data_file, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = item["input"]
        output_text = item["output"]
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        outputs = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": outputs["input_ids"].squeeze()
        }

def get_dataloaders(model_name, batch_size=4):
    """Get train, validation, and test dataloaders.
    
    First tries person_3/data/ (Person 3's own downloader output).
    If not found, falls back to combining Person 1's preprocessed paraphrase splits.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if Person 3's own data exists
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "validation.jsonl"
    test_file = DATA_DIR / "test.jsonl"
    
    if train_file.exists() and val_file.exists() and test_file.exists():
        print("[DATA] Using Person 3's own preprocessed data")
        train_dataset = ParaphraseDataset(train_file, tokenizer)
        val_dataset = ParaphraseDataset(val_file, tokenizer)
        test_dataset = ParaphraseDataset(test_file, tokenizer)
    else:
        print("[DATA] Person 3 data not found, falling back to Person 1's splits...")
        train_records, val_records, test_records = _load_from_person1_splits()
        
        if not train_records:
            raise FileNotFoundError(
                "No training data found. Either:\n"
                "  1. Run 'python dataset_downloader.py' in person_3/\n"
                "  2. Or run Person 1's pipeline first: cd person_1 && python run_all.py"
            )
        
        # Save to person_3/data/ so we don't need to rebuild next time
        _save_records(train_records, train_file)
        _save_records(val_records, val_file)
        _save_records(test_records, test_file)
        
        train_dataset = ParaphraseDataset(train_file, tokenizer)
        val_dataset = ParaphraseDataset(val_file, tokenizer)
        test_dataset = ParaphraseDataset(test_file, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=TRAINING_CONFIG["dataloader_num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG["dataloader_num_workers"]
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAINING_CONFIG["dataloader_num_workers"]
    )
    
    return train_loader, val_loader, test_loader, tokenizer


def _load_from_person1_splits():
    """Load paraphrase data from Person 1's preprocessed splits."""
    train_records = []
    val_records = []
    test_records = []
    
    for ds_name in PERSON1_PARAPHRASE_DATASETS:
        for split, target in [("train", train_records), ("val", val_records), ("test", test_records)]:
            path = PERSON1_SPLITS_DIR / ds_name / f"{split}.jsonl"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            # Only keep paraphrase-format records (input/output)
                            if "input" in record and "output" in record:
                                target.append(record)
                print(f"  Loaded {ds_name}/{split}: {sum(1 for _ in open(path)):,} records")
    
    print(f"  Total: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    return train_records, val_records, test_records


def _save_records(records, path):
    """Save records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
