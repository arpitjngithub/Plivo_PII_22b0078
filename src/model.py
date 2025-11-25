# from transformers import AutoModelForTokenClassification
# from labels import LABEL2ID, ID2LABEL


# def create_model(model_name: str):
#     model = AutoModelForTokenClassification.from_pretrained(
#         model_name,
#         num_labels=len(LABEL2ID),
#         id2label=ID2LABEL,
#         label2id=LABEL2ID,
#     )
#     return model
# --- src/model.py (MODIFIED) ---

import torch
from transformers import AutoModelForTokenClassification, DistilBertConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str):
    # Ensure DistilBERT is used for speed optimization
    if "distilbert" not in model_name.lower():
        print(f"Warning: Using '{model_name}'. Consider 'distilbert-base-uncased' for fast CPU inference.")
        # Fallback to default loading if not explicitly DistilBERT
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        return model

    config = DistilBertConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    # --- START OPTIMIZATION: Freeze layers for faster fine-tuning & lower latency ---
    # Freeze the embeddings
    print(f"Freezing embeddings of {model_name}...")
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze the first 2 of 6 transformer layers (layer[0] and layer[1])
    num_frozen_layers = 2
    print(f"Freezing first {num_frozen_layers} transformer layers...")
    for i in range(num_frozen_layers):
        for param in model.distilbert.transformer.layer[i].parameters():
            param.requires_grad = False
    # --- END OPTIMIZATION ---

    return model