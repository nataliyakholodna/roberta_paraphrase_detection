import pytorch_lightning as pl
from transformers import AutoModel, get_cosine_schedule_with_warmup
import torch
import torch.nn.functional as F
import numpy as np


class Classifier_Model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)

        # dropout layer
        self.dropout_layer = torch.nn.Dropout()

        # hidden and output layers
        self.hidden_layer = torch.nn.Linear(self.pretrained_model.config.hidden_size,
                                      256)

        self.output_layer = torch.nn.Linear(256,
                                      1)

        # initialize weights
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

        # loss function
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    # training loop
    def forward(self, input_ids, attention_mask, labels):
        # pass to RoBerta model
        out = self.pretrained_model(input_ids, attention_mask)

        pulled_output = torch.mean(out.last_hidden_state, 1)

        # nn
        res = self.hidden_layer(pulled_output)
        res = self.dropout_layer(res)
        res = F.relu(res)
        logits = self.output_layer(res)

        # loss
        loss = 0
        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.config['num_classes']),
                                      labels.view(-1, self.config['num_classes']))

        return loss, logits

    def training_step(self, batch, batch_idx):
        # unpack dictionary
        # call forward func
        loss, logits = self(**batch)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return {'loss': loss, 'predictions': logits, 'labels': batch['labels']}

    def validation_step(self, batch, batch_idx):
        # unpack dictionary
        loss, logits = self(**batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'predictions': logits, 'labels': batch['labels']}

    def test_step(self, batch, batch_idx):
        # unpack dictionary
        _, logits = self(**batch)
        return logits

    def configure_optimizers (self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'],
                                      weight_decay=self.config['w_decay'])
        total_steps = self.config['train_size'] / self.config['batch_size']
        warmup_steps = np.floor(total_steps * self.config['warmup_percentage'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [scheduler]