import os
import xml.etree.ElementTree as ET
from dc.download.download_utils import write_dataset, download_dataset, \
    create_directory_for_dataset, split_dataset


def download_iu_xray():
    dataset_name = 'iu_xray'

    create_directory_for_dataset(dataset_name)
    download_dataset(dataset_name)

    images_captions = {}
    images_major_tags = {}
    images_auto_tags = {}
    reports_with_images = {}
    text_of_reports = {}

    reports_path = "iu_xray/ecgen-radiology"
    reports = os.listdir(reports_path)
    reports.sort()
    for report in reports:
        tree = ET.parse(os.path.join(reports_path, report))
        root = tree.getroot()
        img_ids = []
        # find the images of the report
        images = root.findall("parentImage")
        # if there aren't any ignore the report
        if len(images) != 0:
            sections = root.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")
            # find impression and findings sections
            for section in sections:
                if section.get("Label") == "FINDINGS":
                    findings = section.text
                if section.get("Label") == "IMPRESSION":
                    impression = section.text

            if impression is not None and findings is not None:
                caption = impression + " " + findings

                # get the MESH tags
                tags = root.find("MeSH")
                major_tags = []
                auto_tags = []
                if tags is not None:
                    major_tags = [t.text for t in tags.findall("major")]
                    auto_tags = [t.text for t in tags.findall("automatic")]

                for image in images:
                    iid = image.get("id") + ".png"
                    images_captions[iid] = caption
                    img_ids.append(iid)
                    images_major_tags[iid] = major_tags
                    images_auto_tags[iid] = auto_tags

                reports_with_images[report] = img_ids
                text_of_reports[report] = caption
        write_dataset(dataset_name, images_captions, images_major_tags, images_auto_tags=images_auto_tags)
        split_dataset(reports_with_images,dataset_name ,text_of_reports)


def download_peir_gross():
    dataset_name = 'peir_gross'
    create_directory_for_dataset(dataset_name)
    image_captions, image_tags = download_dataset(dataset_name)
    write_dataset(dataset_name, image_captions, image_tags)
    split_dataset(image_captions, dataset_name)


def download_imageCLEF():
    dataset_name = 'imageClef'



download_iu_xray()
