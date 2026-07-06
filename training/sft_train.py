import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from dataclasses import dataclass, field
from typing import Optional
import torch, torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import HfArgumentParser, set_seed, TrainingArguments, Trainer
from utils.utils import create_and_prepare_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChosenOnlyDataset(Dataset):
    """Wraps LOGO dataset to expose only chosen branch for SFT."""
    def __init__(self, logo_dataset, max_seq_length):
        self.ds = logo_dataset
        self.max_seq_length = max_seq_length
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]
        ids = row['chosen_input_ids']
        attn = row['chosen_attention_mask']
        labels = row['chosen_labels']
        # Truncate to max_seq_length
        if len(ids) > self.max_seq_length:
            ids = ids[:self.max_seq_length]
            attn = attn[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
        return {'input_ids': ids, 'attention_mask': attn, 'labels': labels}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama-2")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    max_position_embeddings: int = field(default=4096)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=4)
    peft_model_path: Optional[str] = field(default=None)

@dataclass
class SFTTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataset_path: str = field(default="")
    max_seq_length: int = field(default=5120)
    low_rank_training: bool = field(default=True)
    trainable_params: str = field(default="embed, norm")

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, SFTTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    
    logger.info('Loading model...')
    model, tokenizer = create_and_prepare_model(model_args.model_name_or_path, training_args, model_args)
    model.config.use_cache = not training_args.gradient_checkpointing
    
    logger.info('Loading dataset...')
    dsd = load_from_disk(training_args.dataset_path)
    train_ds = ChosenOnlyDataset(dsd['train'], training_args.max_seq_length)
    eval_ds = ChosenOnlyDataset(dsd['test'], training_args.max_seq_length)
    
    logger.info(f'Train: {len(train_ds)}, Eval: {len(eval_ds)}')
    logger.info('Starting SFT training...')
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    logger.info('SFT training complete.')
