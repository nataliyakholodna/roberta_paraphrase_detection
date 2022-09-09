from predict import roberta_prediction
from transformers import AutoTokenizer
import torch
from model import Classifier_Model
from transformers import logging
from flask import Flask, jsonify, request

logging.set_verbosity_error()

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
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    print('request received')
    data = request.get_json()

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


@app.route("/")
def helloworld():
    return "Hi!"


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_roberta = Classifier_Model.load_from_checkpoint('train/weights-50-epochs.ckpt', config=config)
    model_roberta.to(device)

    app.run(host='0.0.0.0', port=8080,  debug=True)

