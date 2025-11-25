
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # use AdamW from torch.optim
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

from dataset import PIIDataset, collate_batch
from labels import LABEL2ID
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--freeze_base", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading tokenizer:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading model...")
    # try to use repo's create_model helper (if present), otherwise fallback
    try:
        model = create_model(args.model_name)
    except Exception:
        print("create_model failed — falling back to AutoModelForTokenClassification")
        num_labels = len(list(LABEL2ID.keys()))
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )

    if args.freeze_base:
        print("Freezing transformer base...")
        for name, p in model.named_parameters():
            # keep classifier parameters trainable; this heuristic assumes classifier names contain 'classifier'
            if "classifier" not in name.lower():
                p.requires_grad = False

    model.to(args.device)

    print("Loading dataset (PIIDataset expects a path string)...")
    # PIIDataset in your repo expects a path (it opens the file), so pass the path strings directly
    train_ds = PIIDataset(args.train, tokenizer, LABEL2ID, max_length=args.max_length)
    dev_ds = PIIDataset(args.dev, tokenizer, LABEL2ID, max_length=args.max_length)

    # pass collate_fn that knows pad_token_id from tokenizer (collate_batch should return tensors)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id, label_pad_id=-100),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id, label_pad_id=-100),
    )

    optim = AdamW(model.parameters(), lr=args.lr)

    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=total_steps
    )

    print("Training started...")
    model.train()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(train_loader):
            # move tensors to device
            for k in list(batch.keys()):
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(args.device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()

            if step % 20 == 0:
                print(f"Step {step}/{len(train_loader)} - Loss: {loss.item():.4f}")

    print("Training complete!")
    print("Saving model to:", args.out_dir)

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print("All done.")


if __name__ == "__main__":
    main()



# import argparse
# import json
# import os
# import torch
# from torch.utils.data import DataLoader
# from torch.optim import AdamW                       # <- use AdamW from torch.optim
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     get_linear_schedule_with_warmup,
# )

# from dataset import PIIDataset
# from labels import LABEL2ID
# from model import create_model


# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_name", default="distilbert-base-uncased")
#     ap.add_argument("--train", default="data/train.jsonl")
#     ap.add_argument("--dev", default="data/dev.jsonl")
#     ap.add_argument("--out_dir", default="out")
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--epochs", type=int, default=3)
#     ap.add_argument("--lr", type=float, default=5e-5)
#     ap.add_argument("--max_length", type=int, default=256)
#     ap.add_argument("--freeze_base", action="store_true")
#     ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     return ap.parse_args()


# def load_jsonl(path):
#     items = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             items.append(json.loads(line))
#     return items


# def main():
#     args = parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     print("Loading tokenizer:", args.model_name)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)

#     print("Loading model...")
#     # create_model should return a huggingface AutoModelForTokenClassification-compatible model
#     # If your repo already has a create_model helper, use it; otherwise fall back to AutoModelForTokenClassification
#     try:
#         model = create_model(args.model_name)
#     except Exception:
#         print("create_model failed — falling back to AutoModelForTokenClassification")
#         num_labels = len(list(LABEL2ID.keys()))
#         model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=num_labels)

#     if args.freeze_base:
#         print("Freezing transformer base...")
#         for name, p in model.named_parameters():
#             # keep classifier parameters trainable
#             if not name.lower().startswith("classifier"):
#                 p.requires_grad = False

#     model.to(args.device)

#     print("Loading dataset...")
#     train_items = load_jsonl(args.train)
#     dev_items = load_jsonl(args.dev)

#     train_ds = PIIDataset(train_items, tokenizer, LABEL2ID, max_length=args.max_length)
#     dev_ds = PIIDataset(dev_items, tokenizer, LABEL2ID, max_length=args.max_length)

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#     dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)

#     optim = AdamW(model.parameters(), lr=args.lr)

#     total_steps = max(1, len(train_loader) * args.epochs)
#     scheduler = get_linear_schedule_with_warmup(
#         optim, num_warmup_steps=0, num_training_steps=total_steps
#     )

#     print("Training started...")
#     model.train()

#     for epoch in range(args.epochs):
#         print(f"\nEpoch {epoch + 1}/{args.epochs}")

#         for step, batch in enumerate(train_loader):
#             for k in batch:
#                 batch[k] = batch[k].to(args.device)

#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 labels=batch["labels"],
#             )
#             loss = outputs.loss

#             loss.backward()
#             optim.step()
#             scheduler.step()
#             optim.zero_grad()

#             if step % 20 == 0:
#                 print(f"Step {step}/{len(train_loader)} - Loss: {loss.item():.4f}")

#     print("Training complete!")
#     print("Saving model to:", args.out_dir)

#     model.save_pretrained(args.out_dir)
#     tokenizer.save_pretrained(args.out_dir)

#     print("All done.")


# if __name__ == "__main__":
#     main()



# import os
# import argparse
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# from dataset import PIIDataset, collate_batch
# from labels import LABELS
# from model import create_model


# def parse_args():
#     ap = argparse.ArgumentParser()\nap.add_argument('--freeze_base', action='store_true', help='freeze transformer base and train only classifier head')
#     ap.add_argument("--model_name", default="distilbert-base-uncased")
#     ap.add_argument("--train", default="data/train.jsonl")
#     ap.add_argument("--dev", default="data/dev.jsonl")
#     ap.add_argument("--out_dir", default="out")
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--epochs", type=int, default=3)
#     ap.add_argument("--lr", type=float, default=5e-5)
#     ap.add_argument("--max_length", type=int, default=256)
#     ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     return ap.parse_args()


# def main():
#     args = parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#     train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)

#     train_dl = DataLoader(
#         train_ds,
#         batch_size=args.batch_size,
#         shuffle=True,
#         collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
#     )

#     model = create_model(args.model_name)\nif getattr(args, 'freeze_base', False):\n    for name, p in model.named_parameters():\n        if not name.startswith('classifier') and not name.startswith('dropout'):\n            p.requires_grad = False
#     model.to(args.device)
#     model.train()

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
#     total_steps = len(train_dl) * args.epochs
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
#     )

#     for epoch in range(args.epochs):
#         running_loss = 0.0
#         for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
#             input_ids = torch.tensor(batch["input_ids"], device=args.device)
#             attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
#             labels = torch.tensor(batch["labels"], device=args.device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             running_loss += loss.item()

#         avg_loss = running_loss / max(1, len(train_dl))
#         print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

#     model.save_pretrained(args.out_dir)
#     tokenizer.save_pretrained(args.out_dir)
#     print(f"Saved model + tokenizer to {args.out_dir}")


# if __name__ == "__main__":
#     main()
