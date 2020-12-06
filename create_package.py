import dtlpy as dl
import os

package_name = 'video-tracker'
project_name = 'video-tracker-test'
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
project.artifacts.upload(filepath=os.path.join('<path to weight>', 'config_davis.json'),
                         package_name=package_name)
project.artifacts.upload(filepath=os.path.join('<path to weight>', 'SiamMask_DAVIS.pth'),
                         package_name=package_name)

##################
# deploy service #
##################
init_input = [
    dl.FunctionIO(name='project_name', type=dl.PackageInputType.JSON, value=project_name),
    dl.FunctionIO(name='package_name', type=dl.PackageInputType.JSON, value=package.name)
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
annotation = dl.annotations.get(annotation_id='')

execution_input = [
    dl.FunctionIO(name='item', type=dl.PackageInputType.ITEM, value={'item_id': item.id}),
    dl.FunctionIO(name='annotation', type=dl.PackageInputType.ANNOTATION,
                  value={'item_id': item.id, 'annotation_id': annotation.id}),
    dl.FunctionIO(name='frame_duration', type=dl.PackageInputType.JSON, value=100)
]

execution = service.execute(execution_input=execution_input, function_name='track_bounding_box')
