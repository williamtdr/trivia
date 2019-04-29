# Pulls noun phrases, selects answer among those, reconstructs in original context
from __future__ import absolute_import, division, print_function
from random import random, sample, randrange
from functools import reduce
import os
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
from requests.exceptions import HTTPError
from pprint import pprint
import itertools

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
numPossible = 0
numImpossible = 0
numPossibleWithRelevantSentences = 0
numIntentionallyImpossible = 0

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
              questionInformationExtraction,
              realAnswers=None):
    global numImpossible, numPossible, numPossibleWithRelevantSentences, numIntentionallyImpossible

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

    def accumulateString(acc, token):
        if type(acc) == dict:
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
    answers = []
    knowledgeBase = {}
    allPotentialSubjects = []

    print("ANALYSIS OF QUESTION:")
    questionPOS.pretty_print()

    print("QUESTION NAMED ENTITIES:")
    print(questionNamedEntities)

    print("QUESTION INFORMATION EXTRACTION:")
    print(questionInformationExtraction)

    if len(questionNamedEntities) > 0:
        allPotentialSubjects += list(map(lambda x: x['text'], questionNamedEntities))

    if len(questionInformationExtraction) > 0:
        allPotentialSubjects += list(map(lambda x: x['subject'], questionInformationExtraction))

    potentialSubjects = []
    for t in questionPOS.subtrees(lambda t: t.label() == 'NP'):
        potentialSubjects.append(t)

    if len(potentialSubjects) > 0:
        for subject in potentialSubjects:
            startingTokenTree = list(filter(lambda token: next(subject.subtrees(lambda t:
                 t.label() == token['pos'] and token['originalText'] in list(t.leaves())), False), questionTokens))
            sequences = reduce(accumulateLongestSequence, startingTokenTree, None)

            if sequences:
                longestSequence = reduce(lambda acc, seq: seq if len(seq) > len(acc) else acc, sequences)
                longestSequenceAsTokens = reduce(accumulateString, longestSequence, "").strip()
                allPotentialSubjects += [ longestSequenceAsTokens ]

    allPotentialSubjects = list(set(allPotentialSubjects))
    allPotentialSubjects.sort(key=len, reverse=True)

    print("ALL POTENTIAL SUBJECTS:")
    print(allPotentialSubjects)
    potentialSentences = []

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

        for answer in potentialAnswers:
            startingTokenTree = list(filter(lambda token: next(answer.subtrees(lambda t:
                t.label() == token['pos'] and token['originalText'] in list(t.leaves())), False), sentTokens))
            sequences = reduce(accumulateLongestSequence, startingTokenTree, None)
            longestSequenceAsTokens = None

            if sequences:
                longestSequence = reduce(lambda acc, seq: seq if len(seq) > len(acc) else acc, sequences)
                longestSequenceAsTokens = reduce(accumulateString, longestSequence, "").strip()

            fullContext = reduce(accumulateString, sentTokens, "").strip()
            try:
                if not longestSequenceAsTokens:
                    raise AssertionError("No noun phrase found in sentence:")

                if not any(entity in fullContext for entity in allPotentialSubjects):
                    raise AssertionError("No named entity not found in sentence:")

                index = context.index(longestSequenceAsTokens)
                answers.append((index, longestSequenceAsTokens))

                if fullContext not in potentialSentences:
                    potentialSentences.append(fullContext)
            except AssertionError as e:
                print(e.args[0])
                print(fullContext)

                break
            except ValueError:
                print("***")
                if longestSequenceAsTokens:
                    print(longestSequenceAsTokens.strip())

                print(context)
                print("couldn't reconstruct original phrase :(")

    print(potentialSentences)
    print("Relevant context judgement: {0}/{1} relevant sentences".format(len(potentialSentences), len(contextPOS)))
    flatKnowledgeBase = list(itertools.chain.from_iterable(knowledgeBase.values()))
    simpleFlatKnowledgeBase = list(map(lambda x: (x['subject'].lower(), x['relation'].lower(), x['object'].lower()), flatKnowledgeBase))
    hasAnswerInKnowledgeBase = any((entity.lower() in x for x in simpleFlatKnowledgeBase)
        for entity in allPotentialSubjects)

    if realAnswers != None:
        if len(realAnswers) > 0:
            if any(realAnswer.lower() in map(lambda x: x[1].lower(), answers) for realAnswer in realAnswers):
                print("Real answer in answers db.")
                numPossible += 1
            elif any((realAnswer.lower() in x.lower() for x in potentialSentences)
                for realAnswer in realAnswers):
                print("Possible with relevant sentence.")
                numPossibleWithRelevantSentences += 1
            else:
                print("Question not answerable with current strategies.")
                numImpossible += 1
        else:
            numIntentionallyImpossible += 1

        print(
            "possible with noun phrases={0}, possible with sentence context={1}, intentionally impossible = {2}, impossible by current standards={3}, total={4}".format(
                numPossible,
                numPossibleWithRelevantSentences,
                numIntentionallyImpossible,
                numImpossible,
                numImpossible + numPossible + numPossibleWithRelevantSentences + numIntentionallyImpossible
        ))

    if not hasAnswerInKnowledgeBase:
        print("Returning false because no answer in knowledge base.")
        return (True, 0, 0)

    answers = list(set(answers))
    print("All potential answers:")
    print(answers)

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
# answer can be provided for obtaining stats about
def eval(context, question, answers=None):
    # todo: dependency tree stuff
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
                     questionInformationExtraction,
                     answers)

# Teardown code:
def stop():
    if config['isManagingServer']:
        server.stop()

parser = CoreNLPParser(url='http://localhost:9000')
