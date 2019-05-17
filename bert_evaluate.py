#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

preprocessor = BertPreprocessor(vocab_file='./bg_cs_pl_ru_cased_L-12_H-768_A-12/vocab.txt',
                                do_lower_case=False,
                                max_seq_length=512)

with tf.Session() as sess:
 
    # Load model
    tf.train.Saver().restore(sess, './bg_cs_pl_ru_cased_L-12_H-768_A-12/bert_model.ckpt')

    # Get predictions
    features = preprocessor(["Bert z ulicy Sezamkowej"])[0]
    
    print(sess.run(bert.sequence_output, feed_dict={input_ids: [features.input_ids],
                                                    input_mask: [features.input_mask],
                                                    token_type_ids: [features.input_type_ids]}))

    features = preprocessor(["Берт", "с", "Улицы", "Сезам"])[0]

    print(sess.run(bert.sequence_output, feed_dict={input_ids: [features.input_ids],
                                                    input_mask: [features.input_mask],
                                                    token_type_ids: [features.input_type_ids]}))
