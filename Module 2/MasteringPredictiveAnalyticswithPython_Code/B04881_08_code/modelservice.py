from flask import Blueprint
main = Blueprint('main', __name__)

import json
from modelfactory import ModelFactory

from flask import Flask, request

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@main.route("/train/<string:input_data>",methods=["POST"])
def train(input_data):
    parsed_parameters = request.json
    model.train(input_data,parsed_parameters)
    return json.dumps("done training")

@main.route("/predict/<string:input_data>")
def predict(input_data):
    parsed_data = [float(e) for e in input_data.split(',')]
    score = model.predict(parsed_data)
    return json.dumps(score)

def prediction_service(model_name,model_parameters,spark_context):
    global model

    model = ModelFactory(model_name,model_parameters,spark_context)    
    service = Flask(__name__)
    service.register_blueprint(main)
    return service
