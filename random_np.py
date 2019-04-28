# Pulls noun phrases, selects answer among those, reconstructs in original context
from __future__ import absolute_import, division, print_function
from random import random, sample, randrange
from functools import reduce
import os
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
from requests.exceptions import HTTPError
from pprint import pprint

# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

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

def extractor(context, contextPOS, contextTokens, question, questionPOS, questionTokens):
    i = -1
    sentOffset = 0
    answers = []

    for sent in contextPOS:
        i += 1
        potentialAnswers = []
        sentTokens = contextTokens[i]

        # sent.pretty_print()

        for t in sent.subtrees(lambda t: t.label() == 'NP'):
            potentialAnswers.append(t)
            # print(t.leaves())

        if len(potentialAnswers) == 0:
            continue

        randomAnswer = sample(potentialAnswers, 1)[0]
        startingTokenTree = list(filter(lambda token: next(randomAnswer.subtrees(lambda t:
                                                                        t.label() == token['pos']
                                                                        and
                                                                        token['originalText'] in list(t.leaves())),
                                                  False), sentTokens))

        def accumulateLongestSequence(acc, token):
            if type(acc) == dict:
                sequenceOne = accumulateLongestSequence(None, acc)

                return accumulateLongestSequence(sequenceOne, token)

            if acc == None:
                return [ [ token ] ]

            lastSequence = acc[-1]
            lastTokenInSequence = lastSequence[-1]

            if lastTokenInSequence['index'] + 1 == token['index']:
                acc[-1].append(token)
            else:
                acc.append([token])

            return acc

        startingIndex = -1
        def accumulateString(acc, token):
            global startingIndex

            if type(acc) == dict:
                if startingIndex == -1:
                    startingIndex = acc['characterOffsetBegin']

                sequenceOne = accumulateString("", acc)

                return accumulateString(sequenceOne, token)

            before, after, text = token['before'], token['after'], token['originalText']

            if acc == None:
                acc = ""

            if acc[-len(before):] != before:
                acc += before

            acc += text
            acc += after

            return acc

        sequences = reduce(accumulateLongestSequence, startingTokenTree, None)
        longestSequenceAsTokens = None
        if sequences:
            longestSequence = reduce(lambda acc, seq: seq if len(seq) > len(acc) else acc, sequences)
            longestSequenceAsTokens = reduce(accumulateString, longestSequence, "").strip()

        try:
            if not longestSequenceAsTokens:
                raise ValueError()

            index = context.index(longestSequenceAsTokens)
            answers.append((index, longestSequenceAsTokens))
        except ValueError:
            print("***")
            if longestSequenceAsTokens:
                print(longestSequenceAsTokens.strip())

            print(context)
            print("couldn't reconstruct original phrase :(")

        sentOffset += sentTokens[-1]['characterOffsetBegin']
        sentOffset += len(sentTokens[-1]['originalText'])

    if len(answers) == 0:
        print("warning: couldn't find an answer!!!")

        return (True, 0, 0)
    else:
        finalAnswer = sample(answers, 1)[0]

        return (False, finalAnswer[0], len(finalAnswer[1]) + finalAnswer[0])

contextCache = {}

# (impossible (bool), starting index, ending index)
def eval(context, question):
    # todo: ner
    questionParsed = parser.api_call(question, properties={
        'outputFormat': 'json',
        'annotators': 'tokenize,parse,ner',
        'tokenizer.ptb3Escaping': 'false'
    })

    if context in contextCache:
        contextParsed = contextCache[context]
    else:
        try:
            contextParsed = parser.api_call(context, properties={
                'outputFormat': 'json',
                'annotators': 'tokenize,parse,ner',
                'tokenizer.ptb3Escaping': 'false'
            })
            contextCache[context] = contextParsed
        except HTTPError:
            print("CoreNLP parse failed, retrying...")

            return eval(context, question)

    questionPOS = None
    questionTokens = None

    for parsed_sent in questionParsed['sentences']:
        questionPOS = parser.make_tree(parsed_sent)
        questionTokens = parsed_sent['tokens']

    contextResponse = list(map(lambda parsed_sent: (parser.make_tree(parsed_sent), parsed_sent['tokens']), contextParsed['sentences']))
    contextPOS = list(map(lambda x: x[0], contextResponse))
    contextTokens = list(map(lambda x: x[1], contextResponse))

    return extractor(context, contextPOS, contextTokens, question, questionPOS, questionTokens)

# Teardown code:
def stop():
    if config['isManagingServer']:
        server.stop()

parser = CoreNLPParser(url='http://localhost:9000')
