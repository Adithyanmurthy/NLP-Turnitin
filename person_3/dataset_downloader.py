"""
Person 3 - Automatic Dataset Downloader
Downloads and preprocesses all required datasets for humanization module
"""
import os
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import random
from config import DATASETS, DATA_DIR, TRAINING_CONFIG

class DatasetDownloader:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.seed = TRAINING_CONFIG["seed"]
        random.seed(self.seed)
        
    def download_all(self):
        """Download and preprocess all datasets"""
        print("=" * 80)
        print("PERSON 3 - DATASET DOWNLOADER")
        print("=" * 80)
        
        all_paraphrase_data = []
        
        # Download each dataset
        for dataset_key, config in DATASETS.items():
            print(f"\n[{dataset_key.upper()}] Downloading...")
            try:
                data = self.download_dataset(dataset_key, config)
                if data:
                    all_paraphrase_data.extend(data)
                    print(f"✓ {dataset_key}: {len(data)} pairs collected")
            except Exception as e:
                print(f"✗ {dataset_key}: Error - {str(e)}")
                print(f"  Continuing with other datasets...")
        
        # Save combined dataset
        print(f"\n[COMBINING] Total pairs collected: {len(all_paraphrase_data)}")
        self.save_combined_dataset(all_paraphrase_data)
        
        print("\n" + "=" * 80)
        print("DATASET DOWNLOAD COMPLETE")
        print("=" * 80)
        
    def download_dataset(self, dataset_key, config):
        """Download and process individual dataset"""
        dataset_format = config["format"]
        
        if dataset_key == "paranmt":
            return self.process_paranmt(config)
        elif dataset_key == "paws":
            return self.process_paws(config)
        elif dataset_key == "qqp":
            return self.process_qqp(config)
        elif dataset_key == "mrpc":
            return self.process_mrpc(config)
        elif dataset_key == "bea_gec":
            return self.process_bea_gec(config)
        elif dataset_key == "sts":
            return self.process_sts(config)
        elif dataset_key == "hc3":
            return self.process_hc3(config)
        
        return []
    
    def process_paranmt(self, config):
        """Process ParaNMT-50M dataset"""
        try:
            # Try to load from HuggingFace
            # Note: ParaNMT-50M might not be directly available, using alternative
            print("  Note: ParaNMT-50M is large. Using PAWS and QQP as alternatives.")
            return []
        except:
            print("  ParaNMT-50M not available, using other paraphrase datasets")
            return []
    
    def process_paws(self, config):
        """Process PAWS dataset"""
        try:
            dataset = load_dataset(config["hf_path"], config["subset"])
            pairs = []
            
            for split in ["train", "validation"]:
                if split in dataset:
                    for item in tqdm(dataset[split], desc=f"  Processing PAWS {split}"):
                        if item["label"] == 1:  # Paraphrases
                            pairs.append({
                                "input": item["sentence1"],
                                "output": item["sentence2"],
                                "source": "paws"
                            })
            
            return pairs
        except Exception as e:
            print(f"  Error loading PAWS: {e}")
            return []
    
    def process_qqp(self, config):
        """Process Quora Question Pairs dataset"""
        try:
            dataset = load_dataset(config["hf_path"], config["subset"])
            pairs = []
            
            for split in ["train", "validation"]:
                if split in dataset:
                    for item in tqdm(dataset[split], desc=f"  Processing QQP {split}"):
                        if item["label"] == 1:  # Duplicate questions (paraphrases)
                            pairs.append({
                                "input": item["question1"],
                                "output": item["question2"],
                                "source": "qqp"
                            })
            
            # Sample if too large
            if len(pairs) > 100000:
                pairs = random.sample(pairs, 100000)
            
            return pairs
        except Exception as e:
            print(f"  Error loading QQP: {e}")
            return []
    
    def process_mrpc(self, config):
        """Process Microsoft Research Paraphrase Corpus"""
        try:
            dataset = load_dataset(config["hf_path"], config["subset"])
            pairs = []
            
            for split in ["train", "validation"]:
                if split in dataset:
                    for item in tqdm(dataset[split], desc=f"  Processing MRPC {split}"):
                        if item["label"] == 1:  # Paraphrases
                            pairs.append({
                                "input": item["sentence1"],
                                "output": item["sentence2"],
                                "source": "mrpc"
                            })
            
            return pairs
        except Exception as e:
            print(f"  Error loading MRPC: {e}")
            return []
    
    def process_bea_gec(self, config):
        """Process BEA-2019 GEC dataset for human imperfections"""
        try:
            dataset = load_dataset(config["hf_path"])
            pairs = []
            
            # Use error-corrected pairs in reverse (correct -> error) for humanization
            for split in dataset:
                if hasattr(dataset[split], "__iter__"):
                    for item in tqdm(list(dataset[split])[:10000], desc=f"  Processing BEA-GEC {split}"):
                        if "original" in item and "corrected" in item:
                            # Reverse: use corrected as input, original as output
                            pairs.append({
                                "input": item["corrected"],
                                "output": item["original"],
                                "source": "bea_gec"
                            })
            
            return pairs
        except Exception as e:
            print(f"  Error loading BEA-GEC: {e}")
            print("  Continuing without BEA-GEC data...")
            return []
    
    def process_sts(self, config):
        """Process STS Benchmark for meaning preservation evaluation"""
        try:
            dataset = load_dataset(config["hf_path"])
            pairs = []
            
            for split in dataset:
                if hasattr(dataset[split], "__iter__"):
                    for item in tqdm(list(dataset[split]), desc=f"  Processing STS {split}"):
                        if "sentence1" in item and "sentence2" in item:
                            score = item.get("score", 0)
                            if score >= 4.0:  # High similarity pairs
                                pairs.append({
                                    "input": item["sentence1"],
                                    "output": item["sentence2"],
                                    "source": "sts"
                                })
            
            return pairs
        except Exception as e:
            print(f"  Error loading STS: {e}")
            return []
    
    def process_hc3(self, config):
        """Process HC3 dataset for AI->Human pairs"""
        try:
            dataset = load_dataset(config["hf_path"])
            pairs = []
            
            for split in dataset:
                if hasattr(dataset[split], "__iter__"):
                    for item in tqdm(list(dataset[split])[:20000], desc=f"  Processing HC3 {split}"):
                        # Use ChatGPT answer as input, human answer as target
                        if "chatgpt_answers" in item and "human_answers" in item:
                            chatgpt_ans = item["chatgpt_answers"]
                            human_ans = item["human_answers"]
                            
                            if isinstance(chatgpt_ans, list) and len(chatgpt_ans) > 0:
                                chatgpt_ans = chatgpt_ans[0]
                            if isinstance(human_ans, list) and len(human_ans) > 0:
                                human_ans = human_ans[0]
                            
                            if chatgpt_ans and human_ans:
                                pairs.append({
                                    "input": chatgpt_ans,
                                    "output": human_ans,
                                    "source": "hc3"
                                })
            
            return pairs
        except Exception as e:
            print(f"  Error loading HC3: {e}")
            return []
    
    def save_combined_dataset(self, data):
        """Save combined dataset with train/val/test splits"""
        if not data:
            print("Warning: No data collected!")
            return
        
        # Shuffle
        random.shuffle(data)
        
        # Split
        total = len(data)
        train_size = int(total * TRAINING_CONFIG["train_split"])
        val_size = int(total * TRAINING_CONFIG["val_split"])
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Save
        splits = {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
        
        for split_name, split_data in splits.items():
            output_file = self.data_dir / f"{split_name}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  Saved {split_name}: {len(split_data)} pairs -> {output_file}")
        
        # Save metadata
        metadata = {
            "total_pairs": total,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "sources": list(set([item["source"] for item in data])),
            "seed": self.seed
        }
        
        metadata_file = self.data_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n  Metadata saved to {metadata_file}")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_all()
