# Pulls noun phrases, selects answer among those, reconstructs in original context
from random import sample
from functools import reduce
import os
from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser
from requests.exceptions import HTTPError
import itertools
import requests
from qanet_integration import getAnswer, setupQANet

# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

config = {
    'isManagingServer': False,
    'debugSentenceElimination': False,
    'coreNLPServerURL': 'http://localhost:9000'
}
parser = CoreNLPParser(url=config['coreNLPServerURL'])
server = None
STANFORD = os.path.join("models", "stanford-corenlp-full-2018-10-05")
contextCache = {}

def setup(manageServerInternally=False):
    global server

    config['isManagingServer'] = manageServerInternally

    if manageServerInternally:
        print("Starting CoreNLP server...")

        server = CoreNLPServer(
            os.path.join(STANFORD, "stanford-corenlp-3.9.2.jar"),
            os.path.join(STANFORD, "stanford-corenlp-3.9.2-models.jar"),
        )
        server.start()
    else:
        try:
            print("Checking connection to CoreNLP server...")

            requests.get(f'{config["coreNLPServerURL"]}/live')
        except BaseException:
            print("Error connecting to CoreNLP instance! Make sure the server is running in the background.")
            print("The relevant command can be found in the README.")

            exit(1)

    setupQANet()


def accumulateLongestSequence(acc, token):
    if type(acc) == dict:
        sequenceOne = accumulateLongestSequence(None, acc)

        return accumulateLongestSequence(sequenceOne, token)

    if acc is None:
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

    if acc is None:
        acc = ""

    if acc[-len(before):] != before:
        acc += before

    acc += text
    acc += after

    return acc


def extractor(context, contextPOS, contextTokens, contextNamedEntities, contextInformationExtraction,
              questionText, questionPOS, questionTokens, questionNamedEntities, questionInformationExtraction,
              realAnswers = None):
    i = -1
    answers = []
    knowledgeBase = {}
    nextGlobalStats = {
        'numImpossible': 0,
        'numPossible': 0,
        'numPossibleWithRelevantSentences': 0,
        'numIntentionallyImpossible': 0,
    }
    allPotentialSubjects = []
    potentialNPSubjects = []
    relevantSentences = []

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

    for t in questionPOS.subtrees(lambda t: t.label() == 'NP'):
        potentialNPSubjects.append(t)

    for subject in potentialNPSubjects:
        startingTokenTree = list(filter(lambda token: next(subject.subtrees(lambda t:
                                                                            t.label() == token['pos'] and token[
                                                                                'originalText'] in list(
                                                                                t.leaves())), False),
                                        questionTokens))
        sequences = reduce(accumulateLongestSequence, startingTokenTree, None)

        if sequences:
            longestSequence = reduce(lambda acc, seq: seq if len(seq) > len(acc) else acc, sequences)
            longestSequenceAsTokens = reduce(accumulateString, longestSequence, "").strip()
            allPotentialSubjects += [longestSequenceAsTokens]

    allPotentialSubjects = list(set(allPotentialSubjects))
    allPotentialSubjects.sort(key=len, reverse=True)

    print("ALL POTENTIAL SUBJECTS:")
    print(allPotentialSubjects)

    for sent in contextPOS:
        i += 1
        potentialAnswers = []
        sentTokens = contextTokens[i]
        sentInformation = contextInformationExtraction[i]
        sentNamedEntities = contextNamedEntities[i]
        fullContext = reduce(accumulateString, sentTokens, "").strip()

        for item in sentInformation:
            object = item['object']
            subject = item['subject']

            if subject in allPotentialSubjects or object in allPotentialSubjects:
                if fullContext not in relevantSentences:
                    relevantSentences.append(fullContext)

            if object in knowledgeBase:
                knowledgeBase[object].append(item)
            else:
                knowledgeBase[object] = [item]

        for item in sentNamedEntities:
            text = item['text']

            if text in allPotentialSubjects:
                if fullContext not in relevantSentences:
                    relevantSentences.append(fullContext)

        # sent.pretty_print()

        for t in sent.subtrees(lambda t: t.label() == 'NP'):
            potentialAnswers.append(t)
        #            print(t.leaves())

        for answer in potentialAnswers:
            startingTokenTree = list(filter(lambda token: next(answer.subtrees(lambda t:
                                                                               t.label() == token['pos'] and token[
                                                                                   'originalText'] in list(t.leaves())),
                                                               False), sentTokens))
            sequences = reduce(accumulateLongestSequence, startingTokenTree, None)
            longestSequenceAsTokens = None

            if sequences:
                longestSequence = reduce(lambda acc, seq: seq if len(seq) > len(acc) else acc, sequences)
                longestSequenceAsTokens = reduce(accumulateString, longestSequence, "").strip()

            try:
                if not longestSequenceAsTokens:
                    raise AssertionError("No noun phrases in sentence, eliminating:")

                if not any(entity.lower() in fullContext.lower() for entity in allPotentialSubjects):
                    raise AssertionError("No named entity found in sentence, eliminating:")

                index = context.index(longestSequenceAsTokens)
                answers.append((index, longestSequenceAsTokens))

                if fullContext not in relevantSentences:
                    relevantSentences.append(fullContext)
            except AssertionError as e:
                if config['debugSentenceElimination']:
                    print(e.args[0])
                    print(fullContext)

                break
            except ValueError:
                print("***")
                if longestSequenceAsTokens:
                    print(longestSequenceAsTokens.strip())

                print(context)
                print("couldn't reconstruct original phrase :(")

    print("RELEVANT CONTEXT JUDGMENT:")
    if len(relevantSentences) == 0:
        print("No relevant sentences.")
    else:
        print(
            f"{len(relevantSentences)}/{len(contextPOS)} relevant sentences. The following will be passed as context:")
        print(*relevantSentences)

    flatKnowledgeBase = list(itertools.chain.from_iterable(knowledgeBase.values()))
    simpleFlatKnowledgeBase = list(map(lambda x: (x['subject'].lower(), x['relation'].lower(), x['object'].lower()),
                                       flatKnowledgeBase))
    hasAnswerInKnowledgeBase = any((entity.lower() in x for x in simpleFlatKnowledgeBase)
                                   for entity in allPotentialSubjects)
    answers = list(set(answers))
    answers = list(filter(lambda x: x[1].lower() not in list(map(lambda y: y.lower(), allPotentialSubjects)), answers))

    print("FEASIBILITY ANALYSIS:")
    if realAnswers is not None:
        if len(realAnswers) > 0:
            if any(realAnswer.lower() in map(lambda x: x[1].lower(), answers) for realAnswer in realAnswers):
                print("Real answer in answers db.")

                nextGlobalStats['numPossible'] += 1
            elif any((realAnswer.lower() in x.lower() for x in relevantSentences) for realAnswer in realAnswers):
                print("Possible with relevant sentence.")

                nextGlobalStats['numPossibleWithRelevantSentences'] += 1
            else:
                print("Question not answerable with current strategies.")

                nextGlobalStats['numImpossible'] += 1
        else:
            nextGlobalStats['numIntentionallyImpossible'] += 1

    if not hasAnswerInKnowledgeBase:
        print("No subjects found in knowledge base.")

        return True, 0, 0, nextGlobalStats

    if len(relevantSentences) == 0:
        print("No relevant context found.")

        return True, 0, 0, nextGlobalStats

    if len(answers) == 0:
        print("No answers found.")

        return True, 0, 0, nextGlobalStats
    else:
        networkAnswer = getAnswer(context, questionText)
        print("NETWORK ANSWERED:")
        print(networkAnswer)
        try:
            index = context.index(networkAnswer)
        except ValueError:
            print("Failed to reconstruct based on faulty tokenization...")
            # todo: don't pick random answer lol
            finalAnswer = sample(answers, 1)[0]

            return False, finalAnswer[0], len(finalAnswer[1]) + finalAnswer[0], nextGlobalStats

        return False, index, len(networkAnswer) + index, nextGlobalStats


# (impossible (bool), starting index, ending index)
# answer can be provided for obtaining stats about
def eval(context, question, answers = None):
    questionPOS, questionTokens, questionNamedEntities, questionInformationExtraction = None, None, None, None

    # todo: dependency tree stuff
    print("Requesting parse information from CoreNLP server...")
    questionParsed = parser.api_call(question, properties={
        'outputFormat': 'json',
        'annotators': 'tokenize,parse,ner,natlog,openie',
        'tokenizer.options': 'ptb3Escaping=false'
    })

    # todo: dump stale items from cache
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

    for parsed_sent in questionParsed['sentences']:
        questionPOS = parser.make_tree(parsed_sent)
        questionTokens = parsed_sent['tokens']
        questionNamedEntities = parsed_sent['entitymentions']
        questionInformationExtraction = parsed_sent['openie']

    contextResponse = list(map(lambda parsed_sent: (parser.make_tree(parsed_sent),
                                                    parsed_sent['tokens'],
                                                    parsed_sent['entitymentions'],
                                                    parsed_sent['openie']), contextParsed['sentences']))
    contextPOS, contextTokens, contextNamedEntities, contextInformationExtraction = \
        list(map(lambda x: x[0], contextResponse)), list(map(lambda x: x[1], contextResponse)), \
        list(map(lambda x: x[2], contextResponse)), list(map(lambda x: x[3], contextResponse))

    return extractor(context, contextPOS, contextTokens, contextNamedEntities, contextInformationExtraction,
                     question, questionPOS, questionTokens, questionNamedEntities, questionInformationExtraction,
                     answers)


# Teardown code:
def stop():
    if config['isManagingServer']:
        server.stop()