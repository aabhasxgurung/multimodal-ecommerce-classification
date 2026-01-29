# model_utils.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer

# ----------------------------
# 1) EDIT THESE PATHS
# ----------------------------
MODEL_SPECS: Dict[str, Dict] = {
    "multimodal_concat": {
        "path": "models/best_multimodal.pt",
        "requires": ["image", "text"],
    },
    "text_only": {
        "path": "models/best_text_only.pt",
        "requires": ["text"],
    },
    "image_only": {
        "path": "models/best_image_only.pt",
        "requires": ["image"],
    },
}

# Put your class names in correct order (same as label2id order during training)
# If you don't know yet, leave it empty and you'll get numeric indices.
CLASS_NAMES: List[str] = [
    # "....",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ----------------------------
# 2) Models (match notebook)
# ----------------------------
class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        m = models.resnet18(weights=None)  # no download; weights loaded from state_dict anyway
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, input_ids=None, attention_mask=None):
        feats = self.backbone(image)
        return self.head(feats)


class TextOnlyModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str = TEXT_MODEL_NAME, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image=None, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


class MultiModalConcat(nn.Module):
    def __init__(self, num_classes: int, text_model: str = TEXT_MODEL_NAME, dropout: float = 0.2):
        super().__init__()
        img = models.resnet18(weights=None)
        img_dim = img.fc.in_features
        img.fc = nn.Identity()
        self.img_encoder = img
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.ReLU(),
        )

        self.txt_encoder = AutoModel.from_pretrained(text_model)
        txt_dim = self.txt_encoder.config.hidden_size
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, 256),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        img_feats = self.img_encoder(image)
        img_feats = self.img_proj(img_feats)

        txt_out = self.txt_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = txt_out.last_hidden_state[:, 0]
        txt_feats = self.txt_proj(txt_feats)

        fused = torch.cat([img_feats, txt_feats], dim=1)
        return self.head(fused)


def infer_num_classes_from_state_dict(sd: dict) -> int:
    # Your error shows head.3.weight exists, so use that
    if "head.3.weight" in sd:
        return sd["head.3.weight"].shape[0]
    # fallback: find any 2D weight that looks like a classifier
    candidates = [(k, v) for k, v in sd.items() if k.endswith(".weight") and v.ndim == 2]
    # pick the one with smallest in_features-like size last? We'll just pick max out_features weight
    candidates.sort(key=lambda kv: kv[1].shape[0], reverse=True)
    return candidates[0][1].shape[0]


def load_model(model_key: str) -> nn.Module:
    spec = MODEL_SPECS[model_key]
    state_dict = torch.load(spec["path"], map_location="cpu")
    num_classes = infer_num_classes_from_state_dict(state_dict)

    if model_key == "image_only":
        model = ImageOnlyModel(num_classes=num_classes)
    elif model_key == "text_only":
        model = TextOnlyModel(num_classes=num_classes)
    elif model_key == "multimodal_concat":
        model = MultiModalConcat(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    return model


def _preprocess_image(img: Image.Image) -> torch.Tensor:
    x = img_tf(img.convert("RGB"))
    return x.unsqueeze(0)  # [1,C,H,W]


def _preprocess_text(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(
    model_key: str,
    model: nn.Module,
    img: Optional[Image.Image] = None,
    text: Optional[str] = None,
) -> Tuple[str, float, List[Tuple[str, float]]]:

    if model_key == "image_only":
        x_img = _preprocess_image(img).to(DEVICE)
        logits = model(image=x_img)

    elif model_key == "text_only":
        input_ids, attention_mask = _preprocess_text(text or "")
        logits = model(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))

    elif model_key == "multimodal_concat":
        x_img = _preprocess_image(img).to(DEVICE)
        input_ids, attention_mask = _preprocess_text(text or "")
        logits = model(
            image=x_img,
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
        )

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    probs = F.softmax(logits, dim=1).squeeze(0)
    topk = min(5, probs.shape[0])
    confs, idxs = torch.topk(probs, k=topk)

    pred_idx = int(idxs[0].item())
    pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    pred_conf = float(confs[0].item())

    top_list: List[Tuple[str, float]] = []
    for c, i in zip(confs.tolist(), idxs.tolist()):
        lbl = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
        top_list.append((lbl, float(c)))

    return pred_label, pred_conf, top_list
