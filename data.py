from xml.etree import ElementTree
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

text_path = "raw_data/1.1.text.xml"
class Paper:
    """one paper class"""
    def __init__(self):
        self.title = ""
        self.text_id = ""
        self.abstract_raw = []
        self.abstract_id = []
        self.entity_dict = {}
        self.sent_id = [-1]
                
    def read(self, paper, word_id):
        self.title = paper[0].text
        self.text_id = paper.attrib["id"]
        word_position = 0
        # 生のabstractを作成
        abstract = paper[1]
        if len(abstract.text.split()) != 0:
            self.abstract_raw.extend(abstract.text.split())
            word_position += len(abstract.text.split())
        for entity in abstract:
            if entity.tag == "entity":
                self.abstract_raw.append(entity.text)
                tmp = entity.attrib["id"]
                self.entity_dict[tmp] = word_position
                word_position += 1
            if entity.tail:
                if len(entity.tail.split()) != 0:
                    self.abstract_raw.extend(entity.tail.split())
                    word_position += len(entity.tail.split())
        # 生のabstractをidに変更
        for n, word in enumerate(self.abstract_raw):
            if word[-1] == "." or word[-1] == "?" or word[-1] == "!":
                self.sent_id.append(n)
            if word in word_id:
                self.abstract_id.append(word_id[word])
            else:
                word_id[word] = len(word_id)
                self.abstract_id.append(word_id[word])
        self.sent_id.append(len(self.abstract_raw))

    def make_input(self, relation):
        abstract = self.abstract_id
        abstract[self.entity_dict[relation["left"]]] = 2
        abstract[self.entity_dict[relation["right"]]] = 3
        bos, eos = self.search_sentence(self.entity_dict[relation["left"]], self.entity_dict[relation["right"]])
        return xp.array(abstract[bos:eos], dtype = xp.int32), self.label2id(relation["label"], relation["direction"])
    
    def search_sentence(self, left, right):
        for i in range(len(self.sent_id) - 1):
            if self.sent_id[i] < left and self.sent_id[i + 1] >= left:
                bos = self.sent_id[i] + 1
            if self.sent_id[i] < right and self.sent_id[i + 1] >= right:
                eos = self.sent_id[i+1] + 1
        return bos, eos

    def label2id(self, label, direction): #USAGE_FEED=0, USAGE_REVERSE=1, RESULT_FEED=2, RESULT_REVERSE=3, MODEL-FEATURE_FEED=4, MODEL-FEATURE_REVERSE=5, PART_WHOLE_FEED=6, PART_WHOLE_REVERSE=7, TOPIC_FEED=8, TOPIC_REVERSE=9, COMPARE=10
        label_number = 0
        if label == "RESULT":
            label_number += 2
        if label == "MODEL-FEATURE":
            label_number += 4
        if label == "PART_WHOLE":
            label_number += 6
        if label == "TOPIC":
            label_number += 8
        if direction == "REVERSE":
            label_number += 1
        if label == "COMPARE":
            label_number = 10
        return label_number


### train と validの分け方に悩みます
def data_load(path):
    """texts_data > list[text_class]"""
    tree = ElementTree.parse(path)
    root = tree.getroot()
    texts_dict = {} #{"id": <class>}
    word_id = {"BOS":0, "EOS":1, "LEFT_entity":2, "RIGHT_entity":3} #{"word": id} 初期値はembedを確保するために用意しているだけで意味はないです
    for branch in root:
        paper = Paper()
        paper.read(branch, word_id)
        texts_dict[paper.text_id] = paper
        ##texts.append(paper)
    return texts_dict, word_id


def relation_load(path):
    """relations_data > list[relation_class]"""
    relations = []
    for line in open(path, "r"):
        relation = {}
        label, pair = line.strip().split("(")
        pair = pair.split(",")
        if len(pair) == 2:
            left = pair[0]
            right = pair[1][:-1]
            direction = "FEED"
        if len(pair) == 3:
            left = pair[0]
            right = pair[1]
            direction = pair[2][:-1]
        relation["text_id"] = left.split(".")[0]
        relation["label"] = label
        relation["left"] = left
        relation["right"] = right
        relation["direction"] = direction
        relations.append(relation)
    return relations

