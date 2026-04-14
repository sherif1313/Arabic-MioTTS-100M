#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ يدعم LoRA عبر PEFT
✅ يدعم Full Fine-tuning
✅ توزيع حقيقي للـ batch عبر الـ GPUs
"""
import sys, argparse, yaml, torch, json, gc, os, time, datetime, re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ✅ استيراد PEFT لدعم LoRA
from peft import LoraConfig, get_peft_model

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# === Audio Token Offset for MioCodec ===
AUDIO_TOKEN_OFFSET = 70145  # MioCodec [0-12799] -> TTS [70145-82944]
MIO_CODEC_MAX = 12799

# === Dataset Class ===
class ArabicTTSDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer, max_text_len: int = 256, 
                 max_audio_tokens: int = 600):
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] 🔄 Loading manifest: {manifest_path}")
        start = time.time()
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        load_time = time.time() - start
        print(f"[Rank {rank}] ✅ Loaded {len(self.data):,} samples in {load_time:.1f}s")
        
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_tokens = max_audio_tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        audio_tokens = item['audio_tokens']
        
        prompt = f"Convert text to speech: {text.strip()} <|audio|>"
        token_str = ' '.join([f'<|s_{t}|>' for t in audio_tokens])
        full_sequence = f"{prompt} {token_str}"
        
        inputs = self.tokenizer(
            full_sequence,
            max_length=self.max_text_len + self.max_audio_tokens,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        labels = inputs['input_ids'].clone()
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:, :prompt_len] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'text': text,
            'token_count': len(audio_tokens)
        }

def safe_float(val, default):
    try:
        return float(val) if val is not None else default
    except:
        return default

def safe_int(val, default):
    try:
        return int(val) if val is not None else default
    except:
        return default

def setup_ddp(rank, world_size):
    """إعداد البيئة الموزعة"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """تنظيف البيئة الموزعة"""
    if dist.is_initialized():
        dist.destroy_process_group()

def save_checkpoint(model, tokenizer, output_dir: str, step: int, rank: int, is_lora: bool = False):
    """Save checkpoint"""
    if rank == 0:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # التعامل مع DDP و PEFT
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # ✅ حفظ LoRA
        if is_lora:
            model_to_save.save_pretrained(checkpoint_dir)
            print(f"[Rank {rank}] 💾 Saved LoRA Adapter: {checkpoint_dir}")
        else:
            # حفظ كامل النموذج
            model_to_save.save_pretrained(checkpoint_dir, safe_serialization=True)
            print(f"[Rank {rank}] 💾 Saved Full Model: {checkpoint_dir}")
            
        tokenizer.save_pretrained(checkpoint_dir)
    
    if dist.is_initialized():
        dist.barrier(timeout=datetime.timedelta(minutes=30))

def main():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"🚀 Fine-tuning DDP | Rank: {rank}/{world_size} | Local Rank: {local_rank}")
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    device = torch.device(f'cuda:{local_rank}')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/arabic_full_ddp")
    parser.add_argument("--resume-step", type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train_cfg = config # الوصول المباشر لأن الملف مسطح
    
    precision = train_cfg.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    if rank == 0:
        print(f"   Device: CUDA (DDP) | Precision: {dtype}")
        print(f"   Output: {args.output_dir}")
        print("-" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if rank == 0:
        print(f"🔄 Loading base model: {train_cfg.get('base_model_path')}...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        train_cfg["base_model_path"], 
        trust_remote_code=True, 
        torch_dtype=dtype,
        cache_dir=os.path.expanduser("~/.cache/huggingface")
    )
    
    # تحريك النموذج للـ GPU قبل تطبيق LoRA
    base_model = base_model.to(device)

    # === تطبيق LoRA أو Full Training ===
    use_lora = train_cfg.get('lora_enabled', False) or train_cfg.get('use_lora', False)
    
    if use_lora:
        if rank == 0:
            print(f"🔥 Enabling LoRA Fine-tuning...")
        lora_config = LoraConfig(
            r=train_cfg.get("lora_r", 128),
            lora_alpha=train_cfg.get("lora_alpha", 256),
            target_modules=train_cfg.get("lora_target_modules", []),
            lora_dropout=train_cfg.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
        if rank == 0:
            model.print_trainable_parameters()
    else:
        if rank == 0:
            print(f"🔥 Full Fine-tuning (All Params)...")
        for param in base_model.parameters():
            param.requires_grad = True
        model = base_model
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"✅ Trainable params: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True
        )
    
    if rank == 0:
        print(f"🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        train_cfg.get("base_model_path"), 
        trust_remote_code=True,
        cache_dir=os.path.expanduser("~/.cache/huggingface")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if rank == 0:
        print(f"🔄 Loading dataset...")
    
    dataset = ArabicTTSDataset(
        manifest_path=train_cfg.get("manifest_path"),
        tokenizer=tokenizer,
        max_text_len=safe_int(train_cfg.get("max_text_len"), 256),
        max_audio_tokens=safe_int(train_cfg.get("max_audio_tokens"), 600)
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=train_cfg.get("seed", 42))
    
    train_loader = DataLoader(
        dataset,
        batch_size=safe_int(train_cfg.get("batch_size"), 2),
        sampler=sampler,
        num_workers=safe_int(train_cfg.get("num_workers"), 4),
        persistent_workers=False,
        prefetch_factor=safe_int(train_cfg.get("dataloader_prefetch_factor"), 2),
        pin_memory=True,
        drop_last=True
    )
    
    base_lr = safe_float(train_cfg.get("learning_rate"), 2e-5)
    # في حالة LoRA، عادة لا نضرب LR بعدد GPUs بنفس الطريقة، لكن سنبقيها للتقسية
    scaled_lr = base_lr * world_size 
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=scaled_lr,
        weight_decay=safe_float(train_cfg.get("weight_decay"), 0.01),
        betas=(0.9, 0.95)
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=safe_int(train_cfg.get("warmup_steps"), 1000),
        num_training_steps=safe_int(train_cfg.get("max_steps"), 30000)
    )
    
    batch_size = safe_int(train_cfg.get("batch_size"), 2)
    grad_accum = safe_int(train_cfg.get("gradient_accumulation_steps"), 16)
    max_steps = safe_int(train_cfg.get("max_steps"), 30000)
    log_every = safe_int(train_cfg.get("log_every"), 50)
    save_every = safe_int(train_cfg.get("save_every"), 200)
    
    if rank == 0:
        print(f"   Batch per GPU: {batch_size}")
        print(f"   Gradient accumulation: {grad_accum}")
        print(f"   Effective batch: {batch_size * grad_accum * world_size}")
        print(f"   Max steps: {max_steps}")
        print("-" * 60)
    
    model.train()
    global_step = args.resume_step
    best_loss = float('inf')
    
    if rank == 0:
        print(f"\n🔄 Starting Training ({'LoRA' if use_lora else 'Full'})...")
    
    try:
        for epoch in range(10):
            sampler.set_epoch(epoch)
            if rank == 0: print(f"\n📅 Epoch {epoch + 1}")
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=(rank != 0))
            
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # Offset Logic
                mio_mask = (labels >= 0) & (labels <= MIO_CODEC_MAX)
                if mio_mask.any():
                    labels = labels.clone()
                    labels[mio_mask] += AUDIO_TOKEN_OFFSET
                
                with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels, 
                        return_dict=True
                    )
                    loss = outputs.loss / grad_accum
                
                loss.backward()
                
                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), safe_float(train_cfg.get("grad_clip_norm"), 1.0))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    if rank == 0 and global_step % log_every == 0:
                        print(f"📊 Step {global_step}: Loss={loss.item()*grad_accum:.4f}")
                    
                    if rank == 0 and global_step % save_every == 0:
                        save_checkpoint(model, tokenizer, args.output_dir, global_step, rank, use_lora)
                        
                        current_loss = loss.item() * grad_accum
                        if current_loss < best_loss:
                            best_loss = current_loss
                            save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best_model"), global_step, rank, use_lora)
                            print(f"🏆 New best! Loss={best_loss:.4f}")
                    
                    if global_step >= max_steps: break
            if global_step >= max_steps: break
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\n⚠️ Interrupted")
            save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "interrupted"), global_step, rank, use_lora)
    
    if rank == 0:
        print(f"\n✅ Training complete!")
        save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final"), global_step, rank, use_lora)
        
        summary = {
            "model": train_cfg.get("base_model_path"),
            "training_type": "lora_finetune_ddp" if use_lora else "full_finetune_ddp",
            "lora_r": train_cfg.get("lora_r") if use_lora else None,
            "final_loss": best_loss,
            "completed_at": datetime.datetime.now().isoformat()
        }
        with open(os.path.join(args.output_dir, "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
    
    if world_size > 1:
        cleanup_ddp()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_ddp()
