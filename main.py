import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import reporter

import data
import mymodel
import time

xp = cuda.cupy
#xp = np

class MyUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, texts_dict):
        super(MyUpdater, self).__init__(
            train_iter, optimizer)
        self.count = 0
        self.texts_dict = texts_dict

    def update_core(self):
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        batch = train_iter.__next__()
        xs, ls = self.myconverter(batch)
        loss = optimizer.target(xs, ls)
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

    def myconverter(self, batch):
        xs = []
        ls = []
        for relation in batch:
            x, l = self.make_input(relation)
            xs.append(x)
            ls.append(l)
        return xs, xp.array(ls, dtype=xp.int32)
        
    def make_input(self, relation):
        return self.texts_dict[relation["text_id"]].make_input(relation)

    
class MyEvaluater(extensions.Evaluator):
    def __init__(self, iterator, target, converter):
        super(MyEvaluater, self).__init__(
            iterator, target, converter=converter)

    def evaluate(self):
        iterator = self._iterators["main"]
        target = self._targets["main"]

        summary = reporter.DictSummary()
        for batch in iterator:
            observation = {}
            with reporter.report_scope(observation):
                xs, ls = self.converter(batch)
                target(xs, ls)
            summary.add(observation)
        return summary.compute_mean()
        


text_path = "raw_data/1.1.text.xml"
relations_path = "raw_data/1.1.relations.txt"


def main():
    texts_dict, word_id = data.data_load(text_path)
    relations_list = data.relation_load(relations_path)

    devel = relations_list[:200]
    del relations_list[:200]
    train = relations_list

    train_iter = iterators.SerialIterator(train, batch_size=10, shuffle=True)
    test_iter = iterators.SerialIterator(devel, batch_size=10, repeat=False, shuffle=False)
    
    m = mymodel.MyChain(len(word_id))
    model = mymodel.Classifier(m)

    gpu_device = 0
    cuda.get_device(gpu_device).use()
    model.to_gpu(gpu_device)
    
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    
    updater = MyUpdater(train_iter, optimizer, texts_dict)
    trainer = training.Trainer(updater, (30, 'epoch'), out='result')
    
    eval_model = model.copy()
    trainer.extend(MyEvaluater(test_iter, eval_model, updater.myconverter))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time()-start)
###test
# for i in range(2000):
#     batch = train_iter.__next__()
#     xs, ls = updater.myconverter(batch)
#     loss = optimizer.target(xs, ls)
#     optimizer.target.cleargrads()
#     loss.backward()
#     optimizer.update()
#     print(loss.data)
