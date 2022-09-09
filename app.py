from predict import roberta_prediction
from transformers import AutoTokenizer
import torch
from model import Classifier_Model
from transformers import logging
logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sent1 = "Bradd Crellin represented BARLA Cumbria on a tour of Australia with 6 other players representing Britain , " \
        "also on a tour of Australia . "
sent2 = 'Bradd Crellin also represented BARLA Great Britain on a tour through Australia on a tour through Australia ' \
        'with 6 other players representing Cumbria . '

config = {
        'model_name': 'roberta-base',
        'dataset': 'PAWS' ,  # ['MRPC', 'Quora', 'PAWS']
        'batch_size': 64,
        'epochs': 100,
        'num_classes': 1,
        'lr': 1.5e-6,
        'warmup_percentage': 0.2,
        'w_decay': 0.001,
}


model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_roberta = Classifier_Model.load_from_checkpoint('train/weights-50-epochs.ckpt', config=config)
model_roberta.to(device)


roberta_prediction(sent1, sent2, tokenizer, model_roberta, device)
