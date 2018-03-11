import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import reporter

xp = cuda.cupy
#xp = np

class MyChain(Chain):
    """ sum model """
    def __init__(self, vocab_size):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, 200)
            self.mid = L.Linear(200, 200)
            self.out = L.Linear(200, 11)
            self.dropout = 0.2

    def __call__(self, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = F.dropout(self.embed(F.concat(xs, axis=0)), self.dropout)
        exs = F.split_axis(ex, x_section, 0, force_tuple=True)
        sum_list = [F.expand_dims(F.sum(ex, 0), 0) for ex in exs]
        sx = F.concat(sum_list, 0)
        return self.out(self.mid(sx))

    
class MiwaChain(Chain):
    """ miwa model """
    def __init__(self, vocab_size):
        super(MiwaChain, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, 200)
            self.bi_lstm = L.NStepBiLSTM(1, 200, 300, 0.2)
            self.mid = L.Linear(3000, 200)
            self.out = L.Linear(200, 11)
            self.dropout = 0.2

    def __call__(self, xs):
        xs = self.bos_eos(xs)
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = F.dropout(self.embed(F.concat(xs, axis=0)), self.dropout)
        exs = F.split_axis(ex, x_section, 0, force_tuple=True)
        hy, cy, ys = self.bi_lstm(hx=None, cx=None, xs=exs)
        ys =F.split_axis(F.concat(ys, axis=0), x_section, 0, force_tuple=True)
        sentences = []
        for n, x in enumerate(xs):
            sentence = []
            left, right = self.entity_place(x)
            sentence.append(F.sum(ys[n][:left], 0))
            sentence.append(ys[n][left])
            if left + 1 == right:  #左と右のentityの間に文字がない場合の処理
                sentence.append(F.sum(ys[n][left:left+2], 0))
            else:
                sentence.append(F.sum(ys[n][left+1:right], 0))
            sentence.append(ys[n][right])
            sentence.append(F.sum(ys[n][right+1:], 0))
            sentences.append(F.concat(sentence, 0))
        sx = F.reshape(F.concat(sentences, 0), [len(xs), 3000])
        return self.out(self.mid(sx))

        
    def bos_eos(self, xs):
        bos = xp.array([0], dtype=xp.int32)
        eos = xp.array([0], dtype=xp.int32)
        bexs = []
        for x in xs:
            bexs.append(xp.hstack([bos, x, eos]))
        return bexs
    
    def entity_place(self, x):
        for n, w in enumerate(x):
            if w == 2:
                left = n
            if w == 3:
                right = n
        return left, right


class LSTMChain(Chain):
    """ lstm model """
    def __init__(self, vocab_size):
        super(LSTMChain, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, 200)
            self.bi_lstm = L.NStepBiLSTM(1, 200, 300, 0.2)
            self.mid = L.Linear(600, 200)
            self.out = L.Linear(200, 11)
            self.dropout = 0.2

    def __call__(self, xs):
        x_len = [len(x) for x in xs]
        x_section = np.cumsum(x_len[:-1])
        ex = F.dropout(self.embed(F.concat(xs, axis=0)), self.dropout)
        exs = F.split_axis(ex, x_section, 0, force_tuple=True)
        hy, cy, ys = self.bi_lstm(hx=None, cx=None, xs=exs)
        ys = F.split_axis(F.concat(ys, axis=0), x_section, 0, force_tuple=True)
        sum_list = [F.expand_dims(F.sum(y, 0), 0) for y in ys]
        sx = F.concat(sum_list, 0)
        return self.out(self.mid(sx))



class Classifier(Chain):
    """ classifier """
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
