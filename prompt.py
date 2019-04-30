import sys
import inquirer
from ie_solution import eval, setup, stop


# ask to load in the context
# options 1. ask question 2. load new context 3. quit the program

def content():
    content = [
      inquirer.Text('context', message="Please input your context")]
    articles = inquirer.prompt(content)['context']
    return articles

def option():
    options = [
        inquirer.List('options',
                message="Choose one option",
                choices=['Ask a question', 'Load new context', 'Add context', 'Quit'],
            ),
    ]
    optionchosen = inquirer.prompt(options)
    return optionchosen

def prompt():
    print("Welcome!")
    contentinput = content()
    opt = option()
#    print ("what is my answer", opt['options'])
    if opt['options']== "Ask a question":
        Q = [inquirer.Text('question', message="Please type your question")]
        question = inquirer.prompt(Q)['question']
        print ("your question is ", question)
        answer = eval(contentinput,question)
        print (answer)
        option()
        prompt()
    elif opt['options'] == "Load new context":
        articles = ""
        content()
        option()
        prompt()
    elif opt['options'] == "Add context":
        newcontent = contentinput + content()
        option()
        prompt()
    elif opt['options'] == "Quit":
        return

prompt()

#print(question)
