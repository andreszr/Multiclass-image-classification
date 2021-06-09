# 06-run-pytorch-data.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
import os

if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/locations'))

    experiment = Experiment(workspace=ws, name='mic-999')

    config = ScriptRunConfig(
        source_directory='./src',
        script='model.py',
        compute_target='cpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount()],
    )
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='monografia-env',
        file_path='./.azureml/multiclass-image-classification.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()

    # Create a model folder in the current directory
    os.makedirs('./model', exist_ok=True)
    run.download_files(prefix='outputs/model', output_directory='./model', append_prefix=False)
    run.register_model(model_name='mic-999', model_path='outputs/model')

    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)