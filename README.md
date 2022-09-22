# PhraseClassification
A phrase multiclass classification solution.

## Instructions:

### Steps to install requirements of the code:

First, create an virtualenv and activate it with the commands:
```
python3 -m venv env
source env/bin/activate
```
Then run the below code to install the src as root folder and install dependencies.
```
pip3 install -e .
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

To run the post request, the user will need to use a software like insomnia.

## Results
Three different models were testes with different pipelines of pre-processment.

In the end, the data were just set to lowercase, normalized and stopwords removed.

The results were:

### Linear SVC:

Accuracy of 51%

Metrics:
```
               precision    recall  f1-score   support

     finanças       1.00      0.30      0.46        20
     educação       1.00      0.58      0.74        48
   indústrias       0.88      0.56      0.68        27
       varejo       0.76      0.44      0.56        36
orgão público       1.00      0.65      0.79        43

    micro avg       0.93      0.53      0.68       174
    macro avg       0.93      0.51      0.65       174
 weighted avg       0.93      0.53      0.67       174
  samples avg       0.57      0.55      0.56       174
```

### Logistic Regression:

Accuracy of: 31%

Metrics:

```
               precision    recall  f1-score   support

     finanças       1.00      0.20      0.33        20
     educação       1.00      0.27      0.43        48
   indústrias       0.92      0.41      0.56        27
       varejo       0.62      0.14      0.23        36
orgão público       1.00      0.51      0.68        43

    micro avg       0.93      0.32      0.47       174
    macro avg       0.91      0.31      0.45       174
 weighted avg       0.91      0.32      0.46       174
  samples avg       0.35      0.33      0.34       174
```

### Naive Bayes:

Accuracy of 64%

Metrics:
```
               precision    recall  f1-score   support

     finanças       0.83      0.50      0.62        20
     educação       0.92      0.71      0.80        48
   indústrias       0.81      0.81      0.81        27
       varejo       0.80      0.56      0.66        36
orgão público       0.85      0.79      0.82        43

    micro avg       0.85      0.69      0.76       174
    macro avg       0.84      0.67      0.74       174
 weighted avg       0.85      0.69      0.76       174
  samples avg       0.73      0.70      0.71       174
```
