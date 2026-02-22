"""
Person 3 - Train Flan-T5-XL for Humanization
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from data_loader import get_dataloaders
from config import MODELS, CHECKPOINTS_DIR, LOGS_DIR, TRAINING_CONFIG
import os

def train_flan_t5():
    """Train Flan-T5-XL model"""
    print("=" * 80)
    print("TRAINING FLAN-T5-XL FOR HUMANIZATION")
    print("=" * 80)
    
    model_config = MODELS["flan_t5"]
    model_name = model_config["hf_path"]
    
    # Load model and tokenizer
    print(f"\n[1/5] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if TRAINING_CONFIG["fp16"] else torch.float32
    )
    # Enable gradient checkpointing to reduce VRAM usage (~40% savings)
    model.gradient_checkpointing_enable()
    
    # Load data
    print(f"\n[2/5] Loading datasets...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        model_name,
        batch_size=model_config["batch_size"]
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Training arguments
    print(f"\n[3/5] Setting up training configuration...")
    output_dir = CHECKPOINTS_DIR / "flan_t5_xl"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=model_config["epochs"],
        per_device_train_batch_size=model_config["batch_size"],
        per_device_eval_batch_size=model_config["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=model_config["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_dir=str(LOGS_DIR / "flan_t5_xl"),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=TRAINING_CONFIG["fp16"],
        gradient_checkpointing=True,
        report_to="tensorboard",
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    print(f"\n[4/5] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print(f"\n[5/5] Starting training...")
    print(f"  Epochs: {model_config['epochs']}")
    print(f"  Learning rate: {model_config['learning_rate']}")
    print(f"  Batch size: {model_config['batch_size']}")
    print(f"  Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print()
    
    trainer.train()
    
    # Save final model
    print(f"\n[SAVING] Saving final model...")
    final_model_path = CHECKPOINTS_DIR / "flan_t5_xl_final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    print(f"\n✓ Model saved to: {final_model_path}")
    print("=" * 80)
    print("FLAN-T5-XL TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    train_flan_t5()
