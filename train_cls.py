import pandas as pd
import numpy as np
from pythainlp import word_tokenize
from ast import literal_eval
from tqdm import tqdm_notebook
from collections import Counter
import re
import emoji
import string
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback
from pythainlp.ulmfit import *

# #when training with augmented set
train_df = pd.read_csv('train_df.csv')
valid_df = pd.read_csv('valid_df.csv')

#test set
test_df = pd.read_csv('test_df.csv')

model_path = 'prachathai_data/'

#lm data
data_lm = load_data(model_path,'prachathai_lm.pkl')
data_lm.sanity_check()

#classification data
#tt = Tokenizer(tok_func = ThaiTokenizer, lang = 'th', pre_rules = pre_rules_th, post_rules=post_rules_th)
#processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
#            NumericalizeProcessor(vocab=data_lm.vocab, max_vocab=60000, min_freq=3)]
#data_cls = (ItemLists(model_path,train=TextList.from_df(train_df, model_path, cols=['body_text'], processor=processor),
#                     valid=TextList.from_df(valid_df, model_path, cols=['body_text'], processor=processor))
#    .label_from_df(list(train_df.columns[1:]))
#    .add_test(TextList.from_df(test_df, model_path, cols=['body_text'], processor=processor))
#    .databunch(bs=50)
#    )
#data_cls.sanity_check()
#data_cls.save('prachathai_cls.pkl')

#just load
data_cls = load_data(model_path,'prachathai_cls.pkl')
data_cls.sanity_check()
print(len(data_cls.vocab.itos))

#model
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
             output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
trn_args = dict(bptt=70, drop_mult=0.5, alpha=2, beta=1, max_len=500)

learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
#load pretrained finetuned model
learn.load_encoder('prachathai_enc')

#train unfrozen
learn.freeze_to(-1)
learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
learn.freeze_to(-3)
learn.fit_one_cycle(10, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7), 
                    callbacks=[SaveModelCallback(learn, every='improvement', monitor='val_loss', name='prachathai_cls')])
#learn.unfreeze()
#learn.fit_one_cycle(10, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7))

#save test results
probs, y_true = learn.get_preds(ds_type = DatasetType.Test, ordered=True)
probs = probs.numpy()
pickle.dump(probs, open(f'{model_path}probs.pkl','wb'))
