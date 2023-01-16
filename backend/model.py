import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_path = 'model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    return model, tokenizer

def predict_relationship(text1, text2, model, tokenizer):
        if len(text1) == 0 or len(text2) == 0:
            return {}
        with torch.inference_mode():
            out = model(**tokenizer(text1, text2, return_tensors='pt').to(model.device))
            proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
        return {v: float(proba[k]) for k, v in model.config.id2label.items()}


def predict_zero_shot(text, label_texts, model, tokenizer, label='entailment', normalize=True):
    if len(text) == 0 or len(label_texts) == 0:
        return {}
    tokens = tokenizer([text] * len(label_texts), label_texts, truncation=True, return_tensors='pt', padding=True)
    with torch.inference_mode():
        result = torch.softmax(model(**tokens.to(model.device)).logits, -1)
    proba = result[:, model.config.label2id[label]].cpu().numpy()
    if normalize:
        proba /= sum(proba)
    return {label_texts[i]: float(proba[i]) for i in range(len(label_texts))}
