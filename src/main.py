
import os, zipfile, shutil
import supervisely as sly
from supervisely.io.fs import get_file_ext, get_file_name_with_ext
from supervisely.io.json import load_json_file
from supervisely.imaging.image import read
import sly_globals as g


def prepare_ann_data(ann_path):

    ann_json = load_json_file(ann_path)

    for curr_data in ann_json.values():
        polygons = []
        for region in curr_data['regions']:
            polygons.append(region['shape_attributes'])

        g.image_name_to_polygon[curr_data['filename']] = polygons


def create_ann(img_path):
    labels = []

    im = read(img_path)
    width = im.shape[1]
    height = im.shape[0]

    polygons = g.image_name_to_polygon[get_file_name_with_ext(img_path)]
    for poly in polygons:
        x_points = poly['all_points_x']
        y_points = poly['all_points_y']
        points = []
        for idx in range(len(x_points)):
            points.append(sly.PointLocation(y_points[idx], x_points[idx]))

        polygon = sly.Polygon(points, interior=[])
        label = sly.Label(polygon, g.obj_class)
        labels.append(label)

    return sly.Annotation(img_size=(height, width), labels=labels)


# def extract_zip(archive_path):
#     if zipfile.is_zipfile(archive_path):
#         with zipfile.ZipFile(archive_path, 'r') as archive:
#             archive.extractall(g.work_dir_path)
#     else:
#         g.logger.warn('Archive cannot be unpacked {}'.format(get_file_name(archive_path)))
#         g.my_app.stop()


@g.my_app.callback("import_tomato_detection")
@sly.timeit
def import_tomato_detection(api: sly.Api, task_id, context, state, app_logger):

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    for ARH_NAME in g.ARH_NAMES:
        archive_path = os.path.join(g.work_dir_path, ARH_NAME)
        api.file.download(g.TEAM_ID, g.ds_path, archive_path)

        try:
            shutil.unpack_archive(archive_path, g.work_dir_path)
        except Exception('Unknown archive format {}'.format(ARH_NAME)):
            g.my_app.stop()

        ds_name = ARH_NAME.split('-')[0]
        if ds_name not in [g.train_folder_name, g.val_folder_name]:
            g.logger.warn('Folder name is {} but it should be \'train\' or \'val\', check your input data'.format(ds_name))
            g.my_app.stop()

        new_dataset = api.dataset.create(new_project.id, ds_name, change_name_if_conflict=True)

        im_anns_path = os.path.join(g.work_dir_path, ds_name)
        images_pathes = [os.path.join(im_anns_path, item_name) for item_name in os.listdir(im_anns_path) if
                         get_file_ext(item_name) == g.image_ext]

        ann_path = os.path.join(im_anns_path, g.annotations_file_name)
        prepare_ann_data(ann_path)

        progress = sly.Progress('Create dataset {}'.format(ds_name), len(images_pathes), app_logger)
        for img_batch in sly.batched(images_pathes, batch_size=g.batch_size):

            img_names = [get_file_name_with_ext(img_path) for img_path in img_batch]
            img_infos = api.image.upload_paths(new_dataset.id, img_names, img_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(img_path) for img_path in img_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_batch))

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.TEAM_ID,
        "WORKSPACE_ID": g.WORKSPACE_ID
    })
    g.my_app.run(initial_events=[{"command": "import_tomato_detection"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)