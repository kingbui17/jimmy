# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:55:55 2024

@author: home
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:21:22 2024

@author: home
"""

#import io
import pandas as pd
#import requests as r
import numpy as np

file_path = 'C:\\Users\\home\\Downloads\\python'
row_to_use_as_header = 1
questions_output = 'questions.csv'
questions_df= pd.read_csv((file_path + '\\' + questions_output), delimiter=',')
df_questions = questions_df.drop(columns=['id', 'qid1', 'qid2'])
data = pd.read_csv('questions.csv')


data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# difference in lengths of two questions
data['diff_len'] = data.len_q1 - data.len_q2
 # character length based features
data['len_char_q1'] = data.question1.apply(lambda x: 
                  len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: 
                  len(''.join(set(str(x).replace(' ', '')))))
 # word length based features
data['len_word_q1'] = data.question1.apply(lambda x: 
                                         len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: 
                                         len(str(x).split()))
 # common words in the two questions
data['common_words'] = data.apply(lambda x: 
                        len(set(str(x['question1'])
                        .lower().split()).intersection(set(str(x['question2'])
                        .lower().split()))), axis=1)
                                                           
fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1', 
        'len_char_q2', 'len_word_q1', 'len_word_q2',     
        'common_words']


from fuzzywuzzy import fuzz
fuzz.QRatio("Why did Trump win the Presidency?", 
"How did Donald Trump win the 2016 Presidential Election")

fuzz.QRatio("How can I start an online shopping (e-commerce) website?", "Which web technology is best suitable for building a big E-Commerce website?")

fuzz.partial_ratio("Why did Trump win the Presidency?", 
"How did Donald Trump win the 2016 Presidential Election")

fuzz.partial_ratio("How can I start an online shopping (e-commerce) website?", "Which web technology is best suitable for building a big E-Commerce website?")

data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: 
                    fuzz.partial_ratio(str(x['question1']), 
                    str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:
          fuzz.partial_token_set_ratio(str(x['question1']), 
          str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: 
          fuzz.partial_token_sort_ratio(str(x['question1']), 
          str(x['question2'])), axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x: 
                   fuzz.token_set_ratio(str(x['question1']), 
                   str(x['question2'])), axis=1) 
data['fuzz_token_sort_ratio'] = data.apply(lambda x: 
                   fuzz.token_sort_ratio(str(x['question1']), 
                   str(x['question2'])), axis=1)

fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio', 
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']

from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy

tfv_q1 = TfidfVectorizer(min_df=3, 
                         max_features=None, 
                         strip_accents='unicode', 
                         analyzer='word', 
                         token_pattern=r'\w{1,}',
                         ngram_range=(1, 2), 
                         use_idf=1, 
                         smooth_idf=1, 
                         sublinear_tf=1,
                         stop_words='english')
tfv_q2 = deepcopy(tfv_q1)

q1_tfidf = tfv_q1.fit_transform(data.question1.fillna(""))
q2_tfidf = tfv_q2.fit_transform(data.question2.fillna(""))
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
svd_q1 = TruncatedSVD(n_components=180)
svd_q2 = TruncatedSVD(n_components=180)
svd_q3= TruncatedSVD(n_components=180)

question1_vectors = svd_q1.fit_transform(q1_tfidf)
question2_vectors = svd_q2.fit_transform(q2_tfidf)

fs3_1 = sparse.hstack((q1_tfidf, q2_tfidf))
tfv = TfidfVectorizer(min_df=3, 
                      max_features=None, 
                      strip_accents='unicode', 
                      analyzer='word', 
                      token_pattern=r'\w{1,}',
                      ngram_range=(1, 2), 
                      use_idf=1, 
                      smooth_idf=1, 
                      sublinear_tf=1,
                      stop_words='english')
 # combine questions and calculate tf-idf
q1q2 = data.question1.fillna("") 
q1q2 += " " + data.question2.fillna("")
fs3_2 = tfv.fit_transform(q1q2)

fs3_3 = np.hstack((question1_vectors, question2_vectors))

tfv_4 = sparse.hstack((q1_tfidf, q2_tfidf))
fs3_4 = svd_q3.fit_transform(tfv_4)

fs3_svd = tfv.fit_transform(q1q2)
fs3_5 = svd_q3.fit_transform(fs3_svd)

import gensim
#import pyemd 
model = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin.gz', binary=True)

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words = set(stopwords.words('english'))
def sent2vec(s, model):  
    M = []
    words = word_tokenize(str(s).lower())

    for word in words:
        #It shouldn't be a stopword
        if word not in stop_words:
            #nor contain numbers
            if word.isalpha():
                #and be part of word2vec
                if word in model:
                    M.append(model[word])
                    M = np.array(M)
                    if len(M) > 0:
                        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    else:
        return np.zeros(300)
    
w2v_q1 = np.array([sent2vec(q, model) 
                   for q in data.question1])
w2v_q2 = np.array([sent2vec(q, model) 
                   for q in data.question2])

from scipy.spatial.distance import cosine, cityblock,jaccard, canberra, euclidean, minkowski, braycurtis
data['cosine_distance'] = [cosine(x,y) 
                           for (x,y) in zip(w2v_q1, w2v_q2)]
data['cityblock_distance'] = [cityblock(x,y) 
                           for (x,y) in zip(w2v_q1, w2v_q2)]
data['jaccard_distance'] = [jaccard(x,y) 
                           for (x,y) in zip(w2v_q1, w2v_q2)]

data['canberra_distance'] = [canberra(x,y) 
                           for (x,y) in zip(w2v_q1, w2v_q2)]
data['euclidean_distance'] = [euclidean(x,y) 
                           for (x,y) in zip(w2v_q1, w2v_q2)]
data['minkowski_distance'] = [minkowski(x,y,3) 
                           for (x,y) in zip(w2v_q1, w2v_q2)]
data['braycurtis_distance'] = [braycurtis(x,y) 
                           for (x,y) in zip(w2v_q1, w2v_q2)]

fs4_1 = ['cosine_distance', 'cityblock_distance', 
         'jaccard_distance', 'canberra_distance', 
         'euclidean_distance', 'minkowski_distance',
         'braycurtis_distance']

w2v = np.hstack((w2v_q1, w2v_q2))

def wmd(s1, s2, model):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

    data['wmd'] = data.apply(lambda x: wmd(x['question1'],x['question2'], model), axis=1)
    model.init_sims(replace=True) 
    data['norm_wmd'] = data.apply(lambda x: wmd(x['question1'],x['question2'], model), axis=1)

fs4_2 = ['wmd', 'norm_wmd']

import gc
import psutil
del([tfv_q1, tfv_q2, tfv, q1q2, 
     question1_vectors, question2_vectors, svd_q1,svd_q2, q1_tfidf, q2_tfidf])
del([w2v_q1, w2v_q2])
del([model])
gc.collect()
psutil.virtual_memory()

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


scaler = StandardScaler()
y = data.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
X = data[fs_1+fs_2+fs3_4+fs4_1+fs4_2]

X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
X = scaler.fit_transform(X)
X = np.hstack((X, fs3_3))

np.random.seed(42)
n_all, _ = y.shape
idx = np.arange(n_all)
np.random.shuffle(idx)
n_split = n_all // 10
idx_val = idx[:n_split]
idx_train = idx[n_split:]
x_train = X[idx_train]
y_train = np.ravel(y[idx_train])

x_val = X[idx_val]
y_val = np.ravel(y[idx_val])

logres = linear_model.LogisticRegression(C=0.1, 
                                 solver='sag', max_iter=1000)
logres.fit(x_train, y_train)
lr_preds = logres.predict(x_val)
log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
print("Logistic regr accuracy: %0.3f" % log_res_accuracy)

params = dict()
params['objective'] = 'binary:logistic'
params['eval_metric'] = ['logloss', 'error']
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_val, label=y_val)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 5000, watchlist, 
                early_stopping_rounds=50, verbose_eval=100)
xgb_preds = (bst.predict(d_valid) >= 0.5).astype(int)
xgb_accuracy = np.sum(xgb_preds == y_val) / len(y_val)
print("Xgb accuracy: %0.3f" % xgb_accuracy)

import zipfile
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
print("TensorFlow version %s" % tf.__version__)

try:
    df = data[['question1', 'question2', 'is_duplicate']]
except:
    df = pd.read_csv('data/quora_duplicate_questions.tsv',                                                  
                                                    sep='\t')
    df = df.drop(['id', 'qid1', 'qid2'], axis=1)
    
df = df.fillna('')
y = df.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
Tokenizer = tf.keras.preprocessing.text.Tokenizer

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


tk = Tokenizer(num_words=200000)
max_len = 40

tk.fit_on_texts(list(df.question1) + list(df.question2))
x1 = tk.texts_to_sequences(df.question1)
x1 = pad_sequences(x1, maxlen=max_len)
x2 = tk.texts_to_sequences(df.question2)
x2 = pad_sequences(x2, maxlen=max_len) 
word_index = tk.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 300), dtype='float32')
 
glove_zip = zipfile.ZipFile('data/glove.840B.300d.zip')
glove_file = glove_zip.filelist[0]
 
f_in = glove_zip.open(glove_file)
for line in tqdm(f_in):
    values = line.split(b' ')
    word = values[0].decode()
    if word not in word_index:
        continue
    i = word_index[word]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_matrix[i, :] = coefs

f_in.close()
glove_zip.close()

def prepare_batches(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i+step])
    return res

def dense(X, size, activation=None):
    he_std = np.sqrt(2 / int(X.shape[1]))
    out = tf.layers.dense(X, units=size, 
                activation=activation,kernel_initializer=\
                tf.random_normal_initializer(stddev=he_std))
    return out

def time_distributed_dense(X, dense_size):
    shape = X.shape.as_list()
    assert len(shape) == 3
    _, w, d = shape
    X_reshaped = tf.reshape(X, [-1, d])
    H = dense(X_reshaped, dense_size, 
                              tf.nn.relu)
    return tf.reshape(H, [-1, w, dense_size])


def conv1d(inputs, num_filters, filter_size, padding='same'):
    he_std = np.sqrt(2 / (filter_size * num_filters))
    out = tf.layers.conv1d(
        inputs=inputs, filters=num_filters, padding=padding,
        kernel_size=filter_size,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(
                                                 stddev=he_std))
    return out
 
def maxpool1d_global(X):
    out = tf.reduce_max(X, axis=1)
    return out


def lstm(X, size_hidden, size_out):
    with tf.variable_scope('lstm_%d'% np.random.randint(0, 100)):
        he_std = np.sqrt(2 / (size_hidden * size_out))
        W = tf.Variable(tf.random_normal([size_hidden, size_out], 
                                                  stddev=he_std))
        b = tf.Variable(tf.zeros([size_out]))
        size_time = int(X.shape[1])
        X = tf.unstack(X, size_time, axis=1)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size_hidden, 
                                                 forget_bias=1.0)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, X, 
                                                 dtype='float32')
        out = tf.matmul(outputs[-1], W) + b
        return out

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4
learning_rate = 0.001

graph = tf.Graph()
graph.seed = 1
 
with graph.as_default():
    place_q1 = tf.placeholder(tf.int32, shape=(None, max_len))
    place_q2 = tf.placeholder(tf.int32, shape=(None, max_len))

place_y = tf.placeholder(tf.float32, shape=(None, 1))
place_training = tf.placeholder(tf.bool, shape=())
 
glove = tf.Variable(embedding_matrix, trainable=False)
q1_glove_lookup = tf.nn.embedding_lookup(glove, place_q1)
q2_glove_lookup = tf.nn.embedding_lookup(glove, place_q2)
 
emb_size = len(word_index) + 1
emb_dim = 300
emb_std = np.sqrt(2 / emb_dim)
emb = tf.Variable(tf.random_uniform([emb_size, emb_dim],
                                             -emb_std, emb_std))
q1_emb_lookup = tf.nn.embedding_lookup(emb, place_q1)
q2_emb_lookup = tf.nn.embedding_lookup(emb, place_q2)
model1 = q1_glove_lookup

model1 = time_distributed_dense(model1, 300)
model1 = tf.reduce_sum(model1, axis=1)
 
model2 = q2_glove_lookup
model2 = time_distributed_dense(model2, 300)
model2 = tf.reduce_sum(model2, axis=1)
 
model3 = q1_glove_lookup
model3 = conv1d(model3, nb_filter, filter_length,padding='valid')
model3 = tf.layers.dropout(model3, rate=0.2,                                                               training=place_training)
model3 = conv1d(model3, nb_filter, filter_length,                                                                               padding='valid')
model3 = maxpool1d_global(model3)
model3 = tf.layers.dropout(model3, rate=0.2,                                                               training=place_training)
model3 = dense(model3, 300)
model3 = tf.layers.dropout(model3, rate=0.2,training=place_training)

model3 = tf.layers.batch_normalization(model3,                                                                training=place_training)
 
model4 = q2_glove_lookup
model4 = conv1d(model4, nb_filter, filter_length,                                                                                padding='valid')
model4 = tf.layers.dropout(model4, rate=0.2,                                                                training=place_training)
model4 = conv1d(model4, nb_filter, filter_length,                                                                                padding='valid')
model4 = maxpool1d_global(model4)
model4 = tf.layers.dropout(model4, rate=0.2,                                                                training=place_training)
model4 = dense(model4, 300)
model4 = tf.layers.dropout(model4, rate=0.2,                                                                training=place_training)
model4 = tf.layers.batch_normalization(model4,                                                                training=place_training)
 
model5 = q1_emb_lookup
model5 = tf.layers.dropout(model5, rate=0.2,                                                                 training=place_training)
model5 = lstm(model5, size_hidden=300, size_out=300)
 
model6 = q2_emb_lookup
model6 = tf.layers.dropout(model6, rate=0.2,                                                                 training=place_training)
model6 = lstm(model6, size_hidden=300, size_out=300)
 
merged = tf.concat([model1, model2, model3, model4, model5,                                                                         model6], axis=1)

merged = tf.layers.batch_normalization(merged,                                                                         training=place_training)

for i in range(5):
        merged = dense(merged, 300, activation=tf.nn.relu)
        merged = tf.layers.dropout(merged, rate=0.2,                                                                     training=place_training)
        merged = tf.layers.batch_normalization(merged,                                                                     training=place_training)
 
merged = dense(merged, 1, activation=tf.nn.sigmoid)
loss = tf.losses.log_loss(place_y, merged)
prediction = tf.round(merged)
accuracy = tf.reduce_mean(tf.cast(tf.equal(place_y,                                                                      prediction), 'float32'))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
# for batchnorm
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):step = opt.minimize(loss)
 
init = tf.global_variables_initializer()
 
session = tf.Session(config=None, graph=graph)
session.run(init)

np.random.seed(1)
 
n_all, _ = y.shape
idx = np.arange(n_all)
np.random.shuffle(idx)

n_split = n_all // 10
idx_val = idx[:n_split]
idx_train = idx[n_split:]
 
x1_train = x1[idx_train]
x2_train = x2[idx_train]
y_train = y[idx_train]
 
x1_val = x1[idx_val]
x2_val = x2[idx_val]
y_val = y[idx_val]

val_idx = np.arange(y_val.shape[0])
val_batches = prepare_batches(val_idx, 5000)

no_epochs = 10

# see https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0

for i in range(no_epochs):
    np.random.seed(i)
    train_idx_shuffle = np.arange(y_train.shape[0])
    np.random.shuffle(train_idx_shuffle)
    batches = prepare_batches(train_idx_shuffle, 384)
   
    progress = tqdm(total=len(batches))
for idx in batches:
    feed_dict = {place_q1: x1_train[idx],
            place_q2: x2_train[idx],
            place_y: y_train[idx],
            place_training: True,
        }
_, acc, l = session.run([step, accuracy, loss],feed_dict)
progress.update(1)
progress.set_description('%.3f / %.3f' % (acc, l))

y_pred = np.zeros_like(y_val)
for idx in val_batches:
        feed_dict = {
            place_q1: x1_val[idx],
            place_q2: x2_val[idx],
            place_y: y_val[idx],
            place_training: False,
        }
        y_pred[idx, :] = session.run(prediction, feed_dict)
 
print('batch %02d, accuracy: %0.3f' % (i, 
                                 np.mean(y_val == y_pred)))

def convert_text(txt, tokenizer, padder):
    x = tokenizer.texts_to_sequences(txt)
    x = padder(x, maxlen=max_len)
    return x  

def evaluate_questions(a, b, tokenizer, padder, pred):
    feed_dict = {
            place_q1: convert_text([a], tk, pad_sequences),
            place_q2: convert_text([b], tk, pad_sequences),
            place_y: np.zeros((1,1)),
            place_training: False,
        }
    return session.run(pred, feed_dict)
    
isduplicated = lambda a, b: evaluate_questions(a, b, tk, pad_sequences, prediction)

a = "Why are there so many duplicated questions on Quora?"
b = "Why do people ask similar questions on Quora multiple times?"

print("Answer: %0.2f" % isduplicated(a, b))

