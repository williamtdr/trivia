# 1. Break out question into its type: who/what/when/where/why/how
# 2. Train CNN to recognize the answer given words, POS tags, collocated words, answer
from __future__ import absolute_import, division, print_function
from random import random, sample, randrange
from functools import reduce
import os
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
from pprint import pprint

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

parser = None
config = {
    'isManagingServer': False
}
STANFORD = os.path.join("models", "stanford-corenlp-full-2018-10-05")
server = CoreNLPServer(
   os.path.join(STANFORD, "stanford-corenlp-3.9.2.jar"),
   os.path.join(STANFORD, "stanford-corenlp-3.9.2-models.jar"),
)

def setup(manageServerInternally):
    config['isManagingServer'] = manageServerInternally

    if manageServerInternally:
        print("Starting CoreNLP server...")
        server.start()

def whoParser(context, contextPOS, contextTokens, question, questionPOS, questionTokens):
    i = -1
    sentOffset = 0
    answers = []

    #todo: characterOffsetBegin is unreliable, returning a number too high most of the type
    #todo: look into configuration options

    for sent in contextPOS:
        i += 1
        potentialAnswers = []
        sentTokens = contextTokens[i]

        sent.pretty_print()

        for t in sent.subtrees(lambda t: t.label() == 'NP'):
            potentialAnswers.append(t)

        randomAnswer = sample(potentialAnswers, 1)[0]
        startingTokenTree = list(filter(lambda token: next(randomAnswer.subtrees(lambda t:
                                                                        t.label() == token['pos']
                                                                        and
                                                                        token['originalText'] in list(t.leaves())),
                                                  False), sentTokens))

        startingToken = " ".join(randomAnswer.leaves())
        print(startingToken)
        print(startingTokenTree[0]['characterOffsetBegin'])
        try:
            index = context.index(startingToken, int(startingTokenTree[0]['characterOffsetBegin'] - 1))
            answers.append((index, startingToken))
        except ValueError:
            print("couldn't reconstruct original phrase :(")

        sentOffset += sentTokens[-1]['characterOffsetBegin']
        sentOffset += len(sentTokens[-1]['originalText'])

    if len(answers) == 0:
        print("warning: couldn't find an answer!!!")

        return (True, 0, 0)
    else:
        finalAnswer = sample(answers, 1)[0]

        return (False, finalAnswer[0], len(finalAnswer[1]) + finalAnswer[0])

def whatParser(contextTokenized, questionTokenized):
    pass

def whenParser(contextTokenized, questionTokenized):
    pass

def whereParser(contextTokenized, questionTokenized):
    pass

def howParser(contextTokenized, questionTokenized):
    pass

# (impossible (bool), starting index, ending index)
def eval(context, question):
    questionParsed = parser.api_call(question, properties={
        'annotators': 'tokenize,ssplit,parse',
        'tokenizer.ptb3Escaping': 'false'
    })
    contextParsed = parser.api_call(context, properties={
        'annotators': 'tokenize,ssplit,parse',
        'tokenizer.ptb3Escaping': 'false'
    })
    questionPOS = None
    questionTokens = None

    for parsed_sent in questionParsed['sentences']:
        questionPOS = parser.make_tree(parsed_sent)
        questionTokens = parsed_sent['tokens']

    context = list(map(lambda parsed_sent: (parser.make_tree(parsed_sent), parsed_sent['tokens']), contextParsed['sentences']))
    contextPOS = list(map(lambda x: x[0], context))
    contextTokens = list(map(lambda x: x[1], context))

    print(contextPOS)

    return whoParser(context, contextPOS, contextTokens, question, questionPOS, questionTokens)

    return (True, 0, 0)

# Teardown code:
def stop():
    if config['isManagingServer']:
        server.stop()

parser = CoreNLPParser(url='http://localhost:9000')
