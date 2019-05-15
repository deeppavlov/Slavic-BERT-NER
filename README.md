# Slavic BERT NER

**BERT** is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then use that model for downstream NLP tasks that we care about (like question answering). For details see original [BERT github](https://github.com/google-research/bert).

The repository contains **Bulgarian**+**Czech**+**Polish**+**Russian** specific:
- [shared BERT model](#slavic-bert)
- [NER model (`PER`, `LOC`, `ORG`, `PRO`, `EVT`)](#slavic-bert)

## Slavic BERT

The Slavic model is the result of transfer from `2018_11_23/multi_cased_L-12_H-768_A-12` Multilingual BERT model to languages of Bulgarian (`bg`), Czech (`cs`), Polish (`pl`) and Russian (`ru`). The fine-tuning was performed with a stratified dataset of `bg`, `cs` and `pl` Wikipedias and `ru` news.

The model format is the same as in the original repository.

*   **[`BERT, Slavic Cased`](http://files.deeppavlov.ai/deeppavlov_data/bg_cs_pl_ru_cased_L-12_H-768_A-12.tar.gz)**:
    4 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    
#### Usage example

The model can be used in any way proposed by the BERT developers.

One approach may be by using pip packages `bert_dp` and `deeppavlov`:

```python
import tensorflow as tf

from bert_dp.modeling import BertConfig, BertModel
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor


bert_config = BertConfig.from_json_file('./bg_cs_pl_ru_cased_L-12_H-768_A-12/bert_config.json')

input_ids = tf.placeholder(shape=(None, None), dtype=tf.int32) 
input_mask = tf.placeholder(shape=(None, None), dtype=tf.int32) 
token_type_ids = tf.placeholder(shape=(None, None), dtype=tf.int32)

bert = BertModel(config=bert_config,
                 is_training=False,
                 input_ids=input_ids,
                 input_mask=input_mask,
                 token_type_ids=token_type_ids,
                 use_one_hot_embeddings=False)
                 
tf.train.Saver().restore(sess, './bg_cs_pl_ru_cased_L-12_H-768_A-12/bert_model.ckpt')

preprocessor = BertPreprocessor(vocab_file='./bg_cs_pl_ru_cased_L-12_H-768_A-12/vocab.txt',
                                do_lower_case=False,
                                max_seq_length=512)

with tf.Session() as sess:
    features = preprocessor(["Bert z ulicy Sezamkowej"])[0]
    
    print(sess.run(bert.sequence_output, feed_dict={input_ids: [features.input_ids],
                                                    input_mask: [features.input_mask],
                                                    token_type_ids: [features.input_type_ids]}))
```

## Slavic NER

Named Entity Recognition (further, **NER**) is a task of recognizing named entities in text, as well as detecting their type.

We used Slavic BERT model as a base to build NER system. First, we feed each input word into WordPiece cased tokenizer and extract the final hidden representation corresponding to the first subtoken in each word. These representations are fed into a classification dense layer over the NER label set. A token-level CRF layer is also added on top.

![BERT NER diagram](bert_ner_diagram.png)

The model was trained on [BSNLP-2019 dataset](http://bsnlp.cs.helsinki.fi/shared_task.html). The pre-trained model can recognize such entities as:

- Persons (PER)
- Locations (LOC)
- Organizations (ORG)
- Products (PRO)
- Events (EVT)

The metrics for all languages and entities on test set are:

| Language     | Tag  | Precision     | Recall       | RPM (Relaxed Partial Matching)    |
|--------------|:----:|:-------------:|:------------:|:---------------------------------:|
| cs           |      | 94.3          | 93.4         | 93.9                              |
| ru           |      | 88.1          | 86.6         | 87.3                              |
| bg           |      | 90.3          | 84.3         | 87.2                              |
| pl           |      | 93.3          | 93.0         | 93.2                              |
|              | PER  | 94.2          | 95.6         | 94.9                              |
|              | LOC  | 96.6          | 96.4         | 96.5                              |
|              | ORG  | 84.3          | 92.1         | 88.0                              |
|              | PRO  | 87.6          | 51.3         | 64.7                              |
|              | EVT  | 39.4          | 27.7         | 93.9                              |
|              |      | **89.8**      | **91.8**     | **90.8**                          |

For detailed description of evaluation method see [BSNLP-2019 Shared Task page](http://bsnlp.cs.helsinki.fi/shared_task.html).
    
#### Usage example

## References

[1] - [Jacob Devlin et all: *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018](https://arxiv.org/abs/1810.04805)

[2] - [Mozharova V., Loukachevitch N.: *Two-stage approach in Russian named entity recognition*, 2016](https://ieeexplore.ieee.org/document/7584769)

[3] - [BSNLP-2019 Shared Task](http://bsnlp.cs.helsinki.fi/shared_task.html)
