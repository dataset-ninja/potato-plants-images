import os
import shutil
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import dir_exists, file_exists, get_file_ext, get_file_name
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(desc=f"Downloading '{file_name_with_ext}' to buffer...", total=fsize) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    ### Function should read local dataset and upload it to Supervisely project, then return project info.###
    dataset_path = "/home/alex/DATASETS/TODO/Potato Plants Images"
    batch_size = 30
    imgs_ext = ".png"
    anns_ext = ".xml"

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        tag = sly.Tag(meta=curr_tag)

        file_name = get_file_name(image_path)

        ann_path = os.path.join(curr_bboxes_path, file_name + anns_ext)

        if file_exists(ann_path):
            tree = ET.parse(ann_path)
            root = tree.getroot()

            objects_content = root.findall(".//object")
            for obj_data in objects_content:
                name = obj_data.find(".//name").text
                if name not in ["healthy", "stressed"]:
                    print(name)
                if name == "st":
                    name = "stressed"
                obj_class = meta.get_obj_class(name)
                bndbox = obj_data.find(".//bndbox")
                top = int(bndbox.find(".//ymin").text)
                left = int(bndbox.find(".//xmin").text)
                bottom = int(bndbox.find(".//ymax").text)
                right = int(bndbox.find(".//xmax").text)

                rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
                label = sly.Label(rect, obj_class)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[tag])

    obj_class_healthy = sly.ObjClass("healthy", sly.Rectangle)
    obj_class_stressed = sly.ObjClass("stressed", sly.Rectangle)
    name_to_class = {"healthy": obj_class_healthy, "stressed": obj_class_stressed}
    train_tag = sly.TagMeta("train", sly.TagValueType.NONE)
    test_tag = sly.TagMeta("test", sly.TagValueType.NONE)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class_healthy, obj_class_stressed], tag_metas=[train_tag, test_tag]
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in os.listdir(dataset_path):
        curr_ds_path = os.path.join(dataset_path, ds_name)
        if dir_exists(curr_ds_path):
            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

            bboxes_folder_name = "Train_Labels_XML"
            curr_tag = train_tag
            for curr_data in ["Train_Images", "Test_Images"]:
                curr_images_path = os.path.join(curr_ds_path, curr_data)
                images_names = os.listdir(curr_images_path)
                if curr_data == "Test_Images":
                    bboxes_folder_name = "Test_Labels_XML"
                    curr_tag = test_tag
                curr_bboxes_path = os.path.join(curr_ds_path, bboxes_folder_name)

                progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

                for img_names_batch in sly.batched(images_names, batch_size=batch_size):
                    images_pathes_batch = [
                        os.path.join(curr_images_path, image_path) for image_path in img_names_batch
                    ]

                    anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]

                    if curr_data == "Test_Images":
                        img_names_batch = [
                            get_file_name(im_name) + "_test" + get_file_ext(im_name)
                            for im_name in img_names_batch
                        ]

                    img_infos = api.image.upload_paths(
                        dataset.id, img_names_batch, images_pathes_batch
                    )
                    img_ids = [im_info.id for im_info in img_infos]

                    api.annotation.upload_anns(img_ids, anns_batch)

                    progress.iters_done_report(len(img_names_batch))

    return project
