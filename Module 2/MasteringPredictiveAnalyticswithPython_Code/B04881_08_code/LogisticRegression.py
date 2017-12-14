from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import json
from dataparser import DataParser

class LogisticRegression:

	def __init__(self,parameters,sc):
		
		parameters = json.loads(parameters)
		schema = parameters.get('schema',None)
		header = parameters.get('header',False)
		self._parser = DataParser(schema,header)
		self._sc = sc

	def predict(self,input_data):
		return self._model.predict(input_data)

	def train(self,input_data,parameters):
		iterations = parameters.get('iterations',None)
		weights = parameters.get('weights',None)
		intercept = parameters.get('intercept',None)
		numFeatures = parameters.get('numFeatures',None)
		numClasses = parameters.get('numClasses',None)
		data = self._sc.parallelize(self._parser.parse(input_data))
		self._model = LogisticRegressionWithLBFGS.train(data,\
			iterations=iterations,\
			numClasses=numClasses)