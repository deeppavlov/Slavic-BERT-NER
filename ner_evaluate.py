#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deeppavlov import build_model


ner = build_model("./ner_bert_slav.json", download=True)

print(ner(["Bert z ulicy Sezamkowej"]))
# [['Bert z ulicy Sezamkowej'], [['O', 'O', 'B-LOC', 'I-LOC']]]

print(ner(["Берт", "с", "Улицы", "Сезам"]))
# [['Берт', 'с', 'Улицы', 'Сезам'], [['B-PER'], ['O'], ['B-PER'], ['I-PER']]]
