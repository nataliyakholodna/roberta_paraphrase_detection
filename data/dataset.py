from torch.utils.data import Dataset
import pandas as pd
import torch


class Pairs_Dataset(Dataset):
    def __init__(self, data_path, tokenizer,
                 y_label_column, first_message_column, second_message_column,
                 max_token_length=128):

        # init variables
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.y_label_column = y_label_column
        self.first_message_column = first_message_column
        self.second_message_column = second_message_column
        self.max_token_length = max_token_length

        # prepare data at obj initialization
        if 'msr-paraphrase-corpus' or 'paws' in self.data_path:
            self.df = pd.read_csv(data_path, sep='\t', quoting=csv.QUOTE_NONE)
        else:
            self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        current_row = self.df.iloc[index]

        first_message = str(current_row[self.first_message_column])
        second_message = str(current_row[self.second_message_column])

        y = current_row[self.y_label_column]
        label = torch.tensor(y, dtype=torch.float32)

        # {'input_ids': tensor([[101, 487, 1663, 111, 102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]])}
        tokens_dict = self.tokenizer(first_message, second_message,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_token_length,
                                     return_tensors='pt',
                                     add_special_tokens=True)

        return {'input_ids': tokens_dict['input_ids'].flatten(),
                'attention_mask': tokens_dict['attention_mask'].flatten(),
                'labels': label}