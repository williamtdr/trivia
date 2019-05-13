import sys
import PyInquirer 
from ie_solution import eval, setup, stop
from termcolor import colored
from PyInquirer import prompt, print_json

# ask to load in the context
# options 1. ask question 2. load new context 3. quit the program

contentInput = ""

def content():
    global contentInput

    articles = prompt([
        {
            'type': 'input',
            'name': 'context',
            'message': 'Please input your context:'
        }
    ])
    
    if articles == None:
        exit(0)

    articles = articles['context']
    contentInput = articles

    return articles

def option():
    options = [
        {
            'type': 'list',
            'name': 'options',
            'message': 'What do you want to do next?',
            'choices': ['Ask a question', 'Load new context', 'Add context', 'Quit']
        }
    ]
    optionchosen = prompt(options)
    return optionchosen

def menu():
    global contentInput
    opt = option()

    if opt == None:
        exit(0)

    if opt['options'] == "Ask a question":
        Q = [
            {
                'type': 'input',
                'name': 'question',
                'message': 'Please type your question:'
            }
        ]
        question = prompt(Q)['question']
        answer = eval(contentInput, question)
        if not answer[0]:
            start, end = answer[1], answer[2]
            answer = contentInput[start:end].strip()

            print("***")
            print(
                colored(contentInput[0:start], 'yellow'),
                colored(contentInput[start:end], 'magenta'),
                colored(contentInput[end:], 'yellow'),
                sep=''
            )
            print("***")
            print(f">>> Model predicted: {answer}")
        else:
            print(">>> Model predicted: Question is impossible.")

        menu()
    elif opt['options'] == "Load new context":
        contentInput = ""
        content()
        menu()
    elif opt['options'] == "Add context":
        oldcontent = contentInput
        content()
        contentInput = oldcontent + contentInput
        menu()
    elif opt['options'] == "Quit":
        print("Goodbye!")
        stop()
        exit(0)

        return

setup(False)
print("Welcome!")
content()
menu()

#print(question)
