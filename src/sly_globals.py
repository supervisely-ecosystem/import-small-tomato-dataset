
import os, sys
from pathlib import Path
import supervisely as sly
from collections import defaultdict


my_app = sly.AppService()
api: sly.Api = my_app.public_api

root_source_dir = str(Path(sys.argv[0]).parents[1])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)

TASK_ID = int(os.environ["TASK_ID"])
TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])

logger = sly.logger

ds_path = os.environ.get('modal.state.dsPath')

if not ds_path:
    logger.warn('Path to tar file in Team Files is empty, check your input data')
    my_app.stop()
else:
    files = api.file.list2(TEAM_ID, ds_path)

project_name = 'small_tomato'
work_dir = 'tomato_data'

train_folder_name = 'train'
val_folder_name = 'val'
annotations_file_name = 'via_region_data.json'
image_ext = '.jpg'

batch_size = 30
class_name = 'tomato'

width_field = 'width'
height_field = 'height'

image_name_to_polygon = {}

obj_class = sly.ObjClass(class_name, sly.Polygon)
obj_class_collection = sly.ObjClassCollection([obj_class])

meta = sly.ProjectMeta(obj_classes=obj_class_collection)

storage_dir = my_app.data_dir
work_dir_path = os.path.join(storage_dir, work_dir)
sly.io.fs.mkdir(work_dir_path)
