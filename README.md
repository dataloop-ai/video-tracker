# video-tracker

Dataloop FaaS example for functions that tracks bounding-box annotations.

## Download package artifacts

You need to download the artifacts of this package from:
https://storage.googleapis.com/dtlpy/model_assets/video-tracker/artifacts.zip

## SDK Installation

You need to have dtlpy installed, if don't already, install it by running:


```bash
pip install dtlpy --upgrade
```

## Usage

### CLI

```bash
cd <this directory>

dlp projects checkout --project-name <name of the project>

dlp packages push --checkout

dlp packages deploy --checkout
```
### SDK

```python
import dtlpy as dl
import os

package_name = 'video-tracker'
project_name = 'My project'
project = dl.projects.get(project_name=project_name)

##########################
# define package modules #
##########################
modules = [
    dl.PackageModule(
        init_inputs=[
            dl.FunctionIO(name='project_name', type=dl.PackageInputType.JSON),
            dl.FunctionIO(name='package_name', type=dl.PackageInputType.JSON)
        ],
        name='default_module',
        entry_point='main.py',
        functions=[
            dl.PackageFunction(
                inputs=[
                    dl.FunctionIO(name='item', type=dl.PackageInputType.ITEM),
                    dl.FunctionIO(name='annotation', type=dl.PackageInputType.ANNOTATION),
                    dl.FunctionIO(name='frame_duration', type=dl.PackageInputType.JSON)
                ],
                name='track_bounding_box',
                description='Tracks a bounding box annotation on video')
        ]
    )
]

################
# push package #
################
package = project.packages.push(package_name=package_name, modules=modules)

####################
# upload artifacts #
####################
project.artifacts.upload(filepath=os.path.join('weights', 'config_davis.json'),
                         package_name=package_name)
project.artifacts.upload(filepath=os.path.join('weights', 'SiamMask_DAVIS.pth'),
                         package_name=package_name)

##################
# deploy service #
##################
init_input = [
    dl.FunctionIO(name=project_name, type=dl.PackageInputType.JSON, value=project_name),
    dl.FunctionIO(name=package_name, type=dl.PackageInputType.JSON, value=package.name)
]
service = package.services.deploy(service_name=package.name,
                                  module_name='default_module',
                                  init_input=init_input,
                                  runtime={
                                      'numReplicas': 1,
                                      'concurrency': 3,
                                      'podType': dl.InstanceCatalog.GPU_K80_S,
                                      # use our docker image or build one of your own
                                      # see Dockerfile for more info
                                      'runnerImage': 'gcr.io/viewo-g/piper/agent/gpu/torch_opencv_4:1.8.16.0'
                                  })

###########
# execute #
###########
item = dl.items.get(item_id='')
annotation = dl.Annotation.new(
    annotation_definition=dl.Box(
        top=100,
        left=150,
        bottom=200,
        right=300,
        label='car'
    ),
    item=item
)
annotation = annotation.upload()

execution_input = [
    dl.FunctionIO(name='item', type=dl.PackageInputType.ITEM, value={'item_id': item.id}),
    dl.FunctionIO(name='annotation', type=dl.PackageInputType.ANNOTATION,
                  value={'item_id': item.id, 'annotation_id': annotation.id}),
    dl.FunctionIO(name='frame_duration', type=dl.PackageInputType.JSON, value=30)
]

execution = service.execute(execution_input=execution_input)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
