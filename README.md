# PhraseClassification
A phrase multiclass classification solution.

## Instructions:

### Steps to install requirements of the code:

Installing pip-tools a tool to make src the root folder of repo.
```
pip3 install pip-tools
```
Then run the below code to install the src as root folder.
```
pip3 install -e .
```
Finally, run the code to install the requirements.
```
pip3 install -r requirements.txt
```

### Steps to run code in train mode:

Go to src folder and run:
```
python3 classification.py
```

This code will run the train process of three models: Linear SVC, Logistic Regression and Naive Bayes.
All of them will be trained using OneVsRestClassifier strategy.

### Steps to run code in api mode:

go to src/api and run:
```
python3 api.py
```

The server will be open in: http://127.0.0.1:8000. The api call type is Post and the adress is '/sentence'.
The full adress to be used is: http://127.0.0.1:8000/sentence.

The body of the requirement must be:

```
{
	"sentence": ["Curso de Técnico em Segurança do Trabalho por 32x R$ 161,03."],
	"model_path": "../models/naive_bayes.sav"
}
```
The sentence can be just one or a list of sentences to be tested following the pattern.
The model_path is the path of the model the user wants to test.
Following the example, it just needs to change the name of the model, since the models are saved in the same folder.


