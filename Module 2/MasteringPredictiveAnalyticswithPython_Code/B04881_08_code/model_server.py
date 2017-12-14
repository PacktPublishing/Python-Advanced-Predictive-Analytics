# dependencies
import cherrypy
from pyspark import SparkConf, SparkContext
from modelservice import prediction_service
import json

def start_spark_context():
    conf = SparkConf().setAppName("prediction-service").set("spark.driver.allowMultipleContexts",True)
    sc = SparkContext(conf=conf, pyFiles=['modelfactory.py', 'modelservice.py'])

    return sc

def run_server(app):

    import paste
    from paste.translogger import TransLogger

    app_ = TransLogger(app)

    cherrypy.tree.graft(app_, '/')

    cherrypy.config.update({
        'engine.autoreload.on': True,
        'log.screen': True,
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0'
    })

    cherrypy.engine.start()
    cherrypy.engine.block()
    
if __name__ == "__main__":

    sc = start_spark_context()
    parameters = json.loads(open('parameters.json').readline())
    service = prediction_service('LogisticRegression',parameters,sc)

    run_server(service)