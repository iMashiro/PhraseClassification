from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

class Model():
    def train(self, x_train, x_test, y_train, y_test):
        for category in self.classes:
            print('Training... ' + str(category))
            self.model.fit(x_train, y_train)
            prediction = self.model.predict(x_test)
            print('Accuracy: ' + str(accuracy_score(y_test, prediction)))

        print(classification_report(y_test, prediction, target_names=self.classes))