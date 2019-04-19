# Network evaluation script

import json
from random_solution import eval, setup, stop

USE_PRODUCTION_DATA = False
MANAGE_CORENLP_INTERNALLY = False

def evaluate(topics):
    rightTotal = 0
    wrongTotal = 0

    for topic in topics:
        title = topic['title']
        rightTopic = 0
        wrongTopic = 0

        print("Start topic " + title + "!")
        for passage in topic['paragraphs']:
            context = passage['context']
            questions = passage['qas']
            rightPassage = 0
            wrongPassage = 0

            print('***')
            print(context)
            print('***')

            def right(answer):
                print("right! predicted: " + answer)

            def wrong(answer, isActuallyImpossible, evalThinksImpossible, correctAnswer = None):
                if isActuallyImpossible and not evalThinksImpossible:
                    return print("wrong. question is impossible. predicted: " + answer)

                if answer == '' and not isActuallyImpossible:
                    return print("wrong. question is fine.")

                if correctAnswer != None:
                    return print("wrong. predicted: " + answer + ", correct: " + correctAnswer)
                else:
                    return print("wrong. predicted: " + answer)

            for questionObj in questions:
                question = questionObj['question']
                isImpossible = questionObj['is_impossible']
                trainingAnswers = None
                if 'answers' in questionObj:
                    trainingAnswers = questionObj['answers']

                print(question)

                evalIsImpossible, evalStartIndex, evalEndIndex = eval(context, question)
                evalAnswer = context[evalStartIndex:evalEndIndex].strip()
                lastTrainingAnswer = None

                if evalIsImpossible != isImpossible:
                    wrongPassage += 1
                    wrongTopic += 1
                    wrongTotal += 1
                    wrong(evalAnswer, isImpossible, evalIsImpossible)
                else:
                    foundCorrectAnswer = False
                    for potentialSolutionObj in trainingAnswers:
                        start = potentialSolutionObj['answer_start']
                        text = potentialSolutionObj['text']
                        lastTrainingAnswer = text

                        if foundCorrectAnswer:
                            continue

                        # start == evalStartIndex?
                        if text.strip() == evalAnswer:
                            foundCorrectAnswer = True
                            rightPassage += 1
                            rightTopic += 1
                            rightTotal += 1
                            right(evalAnswer)

                    if not foundCorrectAnswer:
                        wrongPassage += 1
                        wrongTopic += 1
                        wrongTotal += 1
                        wrong(evalAnswer, isImpossible, evalIsImpossible, lastTrainingAnswer)

    return (rightTotal, wrongTotal)

fileName = "squad-train-v2.0.json" if USE_PRODUCTION_DATA else "squad-dev-v2.0.json"
print("Loading SQuAD Dataset (" + ("Training" if USE_PRODUCTION_DATA else "Dev") + ")...")

with open('data/' + fileName) as f:
    data = json.load(f)

print("Preparing model...")
setup(MANAGE_CORENLP_INTERNALLY)

right, wrong = evaluate(data['data'])
print("Evaluation complete. Correct: " + str(right) + "/" + str(right + wrong))

print("Stopping dependent systems...")
stop()