from easynmt import EasyNMT
import pandas as pd
import numpy as np
from utils import *


df = pd.read_csv('data_600.csv', header = None).values
df = df[1:]
model = EasyNMT('opus-mt')
array_of_translated_sentences = translate(model, df[:, 1])   
df = pd.DataFrame(array_of_translated_sentences, index = None)
df.to_csv('translations_600.csv', index = False)

df = pd.read_csv('data_864.csv', header = None).values
df = df[1:]
array_of_translated_sentences = translate(model, df[:, 1])   
df = pd.DataFrame(array_of_translated_sentences, index = None)
df.to_csv('translations_864.csv', index = False)
