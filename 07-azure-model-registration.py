# 07-model-registration-azure.py
from azureml.core import Workspace
from azureml.core import Model
from azureml.core import Run

# run = Run.get_context()

if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')

    model = Model.register(model_name='mic_model.h5',
                           tags={'area': 'udea_training'},
                           model_path='outputs/mic_model.h5',
                           model_framework=Model.Framework.TENSORFLOW,
                        #    model_framework_version='2.0',
                           workspace = ws)

    model = run.register_model(model_name='model', model_path='outputs/model')
    print(model.name, model.id, model.version, sep='\t')