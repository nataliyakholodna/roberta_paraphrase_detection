import torch


def roberta_prediction(s1, s2, tokenizer, model, device):

    tokens_dict = tokenizer(s1, s2, padding='max_length', truncation=True,
                            max_length=128,
                            return_tensors='pt',
                            add_special_tokens=True)

    with torch.no_grad():
        model.eval()
        ids = tokens_dict['input_ids']
        mask = tokens_dict['attention_mask']
        label = torch.tensor(0, dtype=torch.float32)

        ids = ids.to(device)
        mask = mask.to(device)
        label = label.to(device)

        _, outputs = model(ids, mask, label)

        pred = torch.sigmoid(outputs).detach().cpu().numpy()[0][0]
        pred_int = 1 if pred > 0.5 else 0

    return pred, pred_int