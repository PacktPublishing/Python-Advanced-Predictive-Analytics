from pyspark.mllib.regression import LabeledPoint

class DataParser:
    
    def __init__(self,schema_dict,header=False):

        '''
        schema takes the form {
        label: dict with position, values (also a dict with key -> position mapping)
        delimiter: delimiter
        features: array of dicts:
            dict with position, values (also a dict with key -> position mapping)
        }
        schema = { 
            'delimiter': ',',
            'label': { 'pos': 3, 'values': { 'Yes': 1.0, 'No': 0.0 } },
            'features': [
                { 'name': 'Class', 'pos': 0, 'values': {'1st':0, '2nd':1, '3rd':2, 'Crew':3  } },
                { 'name': 'Sex', 'pos': 1, 'values': {'Male':0, 'Female':1} },
                { 'name': 'Age', 'pos': 2, 'values': {'Child':0, 'Adult':1} }
            ]
        }


        '''



        self.schema_dict = schema_dict
        self.header = header

    def parse_line(self,input_line):

        try:
            input_array = input_line.strip().replace('\"','').split(self.schema_dict['delimiter'])

            if self.schema_dict['label'].get('values',None) is not None:
                label = self.schema_dict['label']['values'][input_array[self.schema_dict['label']['pos']]]
            else:
                label = input_array[self.schema_dict['label']['pos']] # just get at this position

            features = []

            for f in self.schema_dict['features']:
                # is categorical? 
                if f.get('values',None) is not None:
                    cat_feature = [ 0 ] * len(f['values'].keys())
                    if input_array[f['pos']]!=len(f['values'].keys()): # 1 hot encoding
                        cat_feature[f['values'][input_array[f['pos']]]] = 1
                    features += cat_feature # numerical
                else:
                    features += input_array[f['pos']]
            
            return LabeledPoint(label,features)

        except:
            print('failed to parse line: '.format(input_line))
            pass

    def parse(self,input_data):

        inputs = open(input_data)

        if self.header:
            inputs.readline()

        return [  self.parse_line(l) for l in inputs.readlines() ]

        inputs.close()


