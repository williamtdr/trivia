# trivia
LING 5801 project component - ML based question answering system

# Setup
1. Download CoreNLP from [here](https://stanfordnlp.github.io/CoreNLP/), and place inside a `models` directory inside the project.
2. From the CoreNLP directory (e.g. `stanford-corenlp-full-2018-10-05`), start the server with `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 300000
`.
3.  Run the model evaluator with `python3 eval.py`.