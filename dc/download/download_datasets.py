import os
import json
import random
import numpy as np
import xml.etree.ElementTree as ET
from os.path import join as path_join, abspath


def _create_directory_for_dataset(dataset_directory='iu_xray'):
    try:
        # Create target Directory
        os.mkdir(dataset_directory)
        os.makedirs(abspath(path_join(dataset_directory, dataset_directory+"_images")))
        print("Directory ", dataset_directory, " Created ")
    except FileExistsError:
        print("Directory ", dataset_directory, " already exists")


def _download_dataset(dataset='iu_xray'):
    if dataset is 'iu_xray':
        os.system("wget -P iu_xray/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz")
        # download reports
        os.system("wget -P iu_xray/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz")
        # unzip
        os.system("tar -xzf ./iu_xray/NLMCXR_png.tgz -C iu_xray/iu_xray_images/")
        os.system("tar -xzf ./iu_xray/NLMCXR_reports.tgz -C iu_xray/")
    elif dataset is 'peir_gross':
        pass
    else:
        #download clef
        pass


def _split_images(reports_images, reports_text, img_keys, filename):
    new_images = {}
    for key in img_keys:
        for img in reports_images[key]:
            new_images[img] = reports_text[key]
    with open(filename, "w") as news_images_file:
        for new_image in new_images:
            news_images_file.write(new_image + "\t" + new_images[new_image])
            news_images_file.write("\n")


def _write_dataset(images_captions, images_major_tags, images_auto_tags, dataset='iu_xray'):
    with open(dataset+"/"+dataset+".tsv", "w") as output_file:
        for image_caption in images_captions:
            output_file.write(image_caption + "\t" + images_captions[image_caption])
            output_file.write("\n")

    # Safer JSON storing
    with open(dataset+"/"+dataset+"+_captions.json", "w") as output_file:
        output_file.write(json.dumps(images_captions))
    with open(dataset+"/"+dataset+".json", "w") as output_file:
        output_file.write(json.dumps(images_major_tags))
    with open(dataset+"/"+dataset+"_auto_tags.json", "w") as output_file:
        output_file.write(json.dumps(images_auto_tags))


def _split_dataset(reports_with_images, text_of_reports):
    # perform a case based split
    random.seed(42)
    keys = list(reports_with_images.keys())
    random.shuffle(keys)

    train_split = int(np.floor(len(reports_with_images) * 0.9))

    train_keys = keys[:train_split]
    test_keys = keys[train_split:]

    train_path = "iu_xray/train_images.tsv"
    test_path = "iu_xray/test_images.tsv"

    _split_images(reports_with_images, text_of_reports, train_keys, train_path)
    _split_images(reports_with_images, text_of_reports, test_keys, test_path)


def download_iu_xray():
    #_create_directory_for_dataset()
    #_download_dataset()

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
        _write_dataset(images_captions, images_major_tags, images_auto_tags, dataset='iu_xray')
        _split_dataset(reports_with_images, text_of_reports)


download_iu_xray()

