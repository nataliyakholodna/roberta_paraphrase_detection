### Objective

This app is used as supplementary component for [paraphrase detection](https://github.com/nataliyakholodna/paraphrase_det) model.
It uses trained Roberta model, [PAWS](https://github.com/google-research-datasets/paws) dataset was used for its fine-tuning.
Weights of the model have size of 1.5GB and are not uploaded to this repo. 

App is based on Flask which allows to get model's prediction via GET-request.


Start the server using Docker:

```
docker pull nknknk3000/roberta_paraphrase:1.0
docker run -p 5000:5000 -t -i nknknk3000/roberta_paraphrase:1.0
```

To get RoBERTa's prediction, use GET-request in the following format:
```Python
response = requests.get(url + '/predict',                   
                        params={'s1': sentence_1, 's2': sentence_2})  
```

Where ```sentence_1```, ```sentence_2``` - not preprocessed strings.

Note: size of the ```roberta_paraphrase:1.0``` container is 5.17GB.

Example of the response:
```Python
{'probability': 0.987,
'prediction': 1}
```

Where ```probability``` is float-type value that denotes the probability of two sentenes being the paraphrases of each other;
```prediction``` is an integer which denotes the result of a binary classification: 1 if the sentences are paraphrases, 0 otherwise.

### Project structure

```Python
.
├── data    
    ├── dataset.py  # Contains user-defined PyTorch Dataset 
    └── *.tsv   # Train/test/dev partinions with PAWS dataset
├── train
    ├── weights-50-epochs.ckpt # Weights of the model
    └── train.ipynb # Notebook for training the model                
├── app.py # Main file with Flask app                    
├── model.py # Model based on PyTorch Lightning module                   
├── predict.py # Prediction function                   
├── test_request.py # File for testing the REST API
└── README.md

```
