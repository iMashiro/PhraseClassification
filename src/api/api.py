import flask
import json
from flask import request
from flask_restful import Resource, Api

from src.classification import Classification

app = flask.Flask(__name__)
api = Api(app)

class ReceiveSentence(Resource):
    def post(self):
        request_data = json.loads(request.data)
        sentence = request_data['sentence']
        model_path = request_data['model_path']

        classification = Classification(model_path=model_path)
        result = classification.run_classification(sentence)
        return result


api.add_resource(ReceiveSentence, '/sentence')

app.run(host='127.0.0.1', port=8000)