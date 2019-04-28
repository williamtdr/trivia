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

def extractor(context,
              contextPOS,
              contextTokens,
              contextNamedEntities,
              contextInformationExtraction,
              question,
              questionPOS,
              questionTokens,
              questionNamedEntities,
              questionInformationExtraction):
    def accumulateLongestSequence(acc, token):
        if type(acc) == dict:
            sequenceOne = accumulateLongestSequence(None, acc)

            return accumulateLongestSequence(sequenceOne, token)

        if acc == None:
            return [[token]]

        lastSequence = acc[-1]
        lastTokenInSequence = lastSequence[-1]

        if lastTokenInSequence['index'] + 1 == token['index']:
            acc[-1].append(token)
        else:
            acc.append([token])

        return acc

    # todo: this is a gross dependency
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

    i = -1
    sentOffset = 0
    answers = []
    knowledgeBase = {}

    print("ANALYSIS OF QUESTION:")
    questionPOS.pretty_print()

    print("QUESTION NAMED ENTITIES:")
    print(questionNamedEntities)

    print("QUESTION INFORMATION EXTRACTION:")
    print(questionInformationExtraction)

    namedEntityOfInterest = None
    if len(questionNamedEntities) > 0:
        namedEntityOfInterest = sample(questionNamedEntities, 1)[0]['text']

    if len(questionInformationExtraction) > 0 and namedEntityOfInterest == None:
        print("WARN: Falling back to information extraction mehtod...")
        ieSubject = sample(questionInformationExtraction, 1)[0]
        namedEntityOfInterest = ieSubject['subject']

    if namedEntityOfInterest == None:
        potentialSubjects = []

        for t in questionPOS.subtrees(lambda t: t.label() == 'NP'):
            potentialSubjects.append(t)

        if len(potentialSubjects) > 0:
            subject = sample(potentialSubjects, 1)[0]
            startingTokenTree = list(filter(lambda token: next(subject.subtrees(lambda t:
                 t.label() == token['pos'] and token['originalText'] in list(t.leaves())), False), questionTokens))
            sequences = reduce(accumulateLongestSequence, startingTokenTree, None)

            if sequences:
                longestSequence = reduce(lambda acc, seq: seq if len(seq) > len(acc) else acc, sequences)
                longestSequenceAsTokens = reduce(accumulateString, longestSequence, "").strip()

                namedEntityOfInterest = longestSequenceAsTokens

    if namedEntityOfInterest == None:
        print("WARN: No named entity of interest!")
    else:
        print("NAMED ENTITY OF INTEREST:")
        print(namedEntityOfInterest)

    for sent in contextPOS:
        i += 1
        potentialAnswers = []
        sentTokens = contextTokens[i]
        sentInformation = contextInformationExtraction[i]

        for item in sentInformation:
            object = item['object']

            if object in knowledgeBase:
                knowledgeBase[object].append(item)
            else:
                knowledgeBase[object] = [ item ]

        # sent.pretty_print()

        for t in sent.subtrees(lambda t: t.label() == 'NP'):
            potentialAnswers.append(t)
#            print(t.leaves())

        if len(potentialAnswers) == 0:
            continue

        randomAnswer = sample(potentialAnswers, 1)[0]
        startingTokenTree = list(filter(lambda token: next(randomAnswer.subtrees(lambda t:
            t.label() == token['pos'] and token['originalText'] in list(t.leaves())), False), sentTokens))

        startingIndex = -1
        sequences = reduce(accumulateLongestSequence, startingTokenTree, None)
        longestSequenceAsTokens = None
        if sequences:
            longestSequence = reduce(lambda acc, seq: seq if len(seq) > len(acc) else acc, sequences)
            longestSequenceAsTokens = reduce(accumulateString, longestSequence, "").strip()
        try:
            if not longestSequenceAsTokens:
                raise AssertionError("No noun phrase found in sentence:")
            # print("KNOWLEDGE BASE:")
            # print(knowledgeBase)

            if namedEntityOfInterest != None and namedEntityOfInterest not in " ".join(sent.leaves()):
                raise AssertionError("Named entity not found in sentence:")

            index = context.index(longestSequenceAsTokens)
            answers.append((index, longestSequenceAsTokens))
        except AssertionError as e:
            print(e.args[0])
            fullContext = reduce(accumulateString, sentTokens, "").strip()

            print(fullContext)
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

        print("FINAL ANSWER:")
        print(finalAnswer)

        return (False, finalAnswer[0], len(finalAnswer[1]) + finalAnswer[0])

contextCache = {}

# (impossible (bool), starting index, ending index)
def eval(context, question):
    # todo: ner, dep extraction
    questionParsed = parser.api_call(question, properties={
        'outputFormat': 'json',
        'annotators': 'tokenize,parse,ner,natlog,openie',
        'tokenizer.options': 'ptb3Escaping=false'
    })

    if context in contextCache:
        contextParsed = contextCache[context]
    else:
        try:
            contextParsed = parser.api_call(context, properties={
                'outputFormat': 'json',
                'annotators': 'tokenize,parse,ner,natlog,openie',
                'tokenizer.options': 'ptb3Escaping=false'
            })
            contextCache[context] = contextParsed
        except HTTPError:
            print("CoreNLP parse failed, retrying...")

            return eval(context, question)

    questionPOS = None
    questionTokens = None
    questionNamedEntities = None
    questionInformationExtraction = None

    for parsed_sent in questionParsed['sentences']:
        questionPOS = parser.make_tree(parsed_sent)
        questionTokens = parsed_sent['tokens']
        questionNamedEntities = parsed_sent['entitymentions']
        questionInformationExtraction = parsed_sent['openie']

    contextResponse = list(map(lambda parsed_sent: (parser.make_tree(parsed_sent),
                                                    parsed_sent['tokens'],
                                                    parsed_sent['entitymentions'],
                                                    parsed_sent['openie']), contextParsed['sentences']))
    contextPOS = list(map(lambda x: x[0], contextResponse))
    contextTokens = list(map(lambda x: x[1], contextResponse))
    contextNamedEntities = list(map(lambda x: x[2], contextResponse))
    contextInformationExtraction = list(map(lambda x: x[3], contextResponse))

    return extractor(context,
                     contextPOS,
                     contextTokens,
                     contextNamedEntities,
                     contextInformationExtraction,
                     question,
                     questionPOS,
                     questionTokens,
                     questionNamedEntities,
                     questionInformationExtraction)

# Teardown code:
def stop():
    if config['isManagingServer']:
        server.stop()

parser = CoreNLPParser(url='http://localhost:9000')
