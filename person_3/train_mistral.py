"""
Person 3 - Train Mistral-7B with QLoRA for Humanization
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_loader import get_dataloaders
from config import MODELS, CHECKPOINTS_DIR, LOGS_DIR, TRAINING_CONFIG

def train_mistral():
    """Train Mistral-7B with QLoRA"""
    print("=" * 80)
    print("TRAINING MISTRAL-7B WITH QLORA FOR HUMANIZATION")
    print("=" * 80)
    
    model_config = MODELS["mistral"]
    model_name = model_config["hf_path"]
    
    # Quantization config for QLoRA
    print(f"\n[1/6] Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model and tokenizer
    print(f"\n[2/6] Loading model: {model_name}")
    
    # Delete corrupted cache files if they exist
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if model_name in repo.repo_id:
                for revision in repo.revisions:
                    for f in revision.files:
                        if "model-00002" in str(f.file_path):
                            print(f"  Deleting corrupted cache: {f.file_path}")
                            f.file_path.unlink(missing_ok=True)
    except Exception:
        pass
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        force_download=True,
        resume_download=False,
    )
    
    # Prepare model for training
    print(f"\n[3/6] Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    print(f"\n[4/6] Configuring LoRA...")
    lora_config = LoraConfig(
        r=model_config["lora_r"],
        lora_alpha=model_config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=model_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print(f"\n[5/6] Loading datasets...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        model_name,
        batch_size=model_config["batch_size"]
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Training arguments
    output_dir = CHECKPOINTS_DIR / "mistral_7b_qlora"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=model_config["epochs"],
        per_device_train_batch_size=model_config["batch_size"],
        per_device_eval_batch_size=model_config["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=model_config["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        logging_dir=str(LOGS_DIR / "mistral_7b_qlora"),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        report_to="tensorboard",
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    print(f"\n[6/6] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {model_config['epochs']}")
    print(f"  Learning rate: {model_config['learning_rate']}")
    print(f"  Batch size: {model_config['batch_size']}")
    print(f"  LoRA rank: {model_config['lora_r']}")
    print(f"  LoRA alpha: {model_config['lora_alpha']}")
    print()
    
    trainer.train()
    
    # Save final model
    print(f"\n[SAVING] Saving final model...")
    final_model_path = CHECKPOINTS_DIR / "mistral_7b_qlora_final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    print(f"\n✓ Model saved to: {final_model_path}")
    print("=" * 80)
    print("MISTRAL-7B QLORA TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    train_mistral()
