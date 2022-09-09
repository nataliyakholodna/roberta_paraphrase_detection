from predict import roberta_prediction
from transformers import AutoTokenizer
import torch
from model import Classifier_Model
from transformers import logging
from flask import Flask, jsonify, request

logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sent1 = "Bradd Crellin represented BARLA Cumbria on a tour of Australia with 6 other players representing Britain , " \
        "also on a tour of Australia . "
sent2 = 'Bradd Crellin also represented BARLA Great Britain on a tour through Australia on a tour through Australia ' \
        'with 6 other players representing Cumbria . '

config = {
    'model_name': 'roberta-base',
    'dataset': 'PAWS',  # ['MRPC', 'Quora', 'PAWS']
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


app = Flask(__name__)


@app.post('/predict')
def predict():
    data = request.json
    try:
        s1 = data['s1']
        s2 = data['s2']
    except KeyError:
        return jsonify({'key': 'incorrect data'})

    proba, prediction = roberta_prediction(s1, s2,
                                           tokenizer,
                                           model_roberta,
                                           device)
    try:
        result = jsonify({'probability': proba, 'prediction': prediction})

    except TypeError as err:
        return jsonify({'error': str(err)})
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
