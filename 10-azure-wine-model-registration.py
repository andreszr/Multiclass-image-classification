# 07-model-registration-azure.py
from azureml.core import Workspace
from azureml.core import Model

if __name__ == "__main__":
    ws = Workspace.from_config()

    model = Model.register(model_name='wine_model',
                           tags={'area': 'udea_training'},
                           model_path='outputs/wine_model.pkl',
                           workspace = ws)
    print(model.name, model.id, model.version, sep='\t')