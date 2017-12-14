import importlib

class ModelFactory:

	def train(self,input_data,parameters):
		'''
		trains a model based on input data
		'''
		self.model.train(input_data,parameters)

	def predict(self,data):
		return self.model.predict(data)

	def __init__(self,model_name,model_parameters,sc):
		'''
		loads a model class and initializes sparkContext
		'''
		module = importlib.import_module(model_name)
		model_class = getattr(module, model_name)
		self.model = model_class(model_parameters,sc)

