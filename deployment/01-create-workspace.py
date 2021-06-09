# tutorial/01-create-workspace.py
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace
from src.config import tenant_id, subscription_id

interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id)
ws = Workspace.create(name='azure-ml',
            subscription_id=subscription_id,
            resource_group='monografia',
            create_resource_group=True,
            location='eastus2',
            auth=interactive_auth
            )
            
# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')
