import os
import json
import random
import requests
import numpy as np
import xml.etree.ElementTree as ElementTree
import bioCaption.configuration as conf
from os.path import join as path_join, abspath
from bs4 import BeautifulSoup
import bioCaption.default_config as config


def download_bio_embeddings(path):
    os.system(
        "wget " + "-P " + path + " https://archive.org/download/pubmed2018_w2v_200D.tar/pubmed2018_w2v_200D.tar.gz")
    # Unzip word embeddings
    os.system("tar xvzf pubmed2018_w2v_200D.tar.gz")
    os.system("rm  pubmed2018_w2v_200D.tar.gz")


class DownloadData:
    _logger = conf.get_logger()

    def __init__(self, dataset_path=config.DOWNLOAD_PATH):
        self.dataset_path = dataset_path

    def download_iu_xray(self, split_rate=0.8):
        """Downloads the iu_xray dataset
        :param split_rate: Percentage of the dataset to be kept as training.
        :return: Writes a folder with the dataset name, in the current working
        directory with the dataset.
        """
        dataset_name = 'iu_xray'

        self._create_directory_for_dataset(dataset_name)
        self._download_dataset(dataset_name)

        images_captions = {}
        images_major_tags = {}
        images_auto_tags = {}
        reports_with_images = {}
        text_of_reports = {}

        reports_path = "iu_xray/ecgen-radiology"
        reports = os.listdir(reports_path)
        reports.sort()
        for report in reports:
            tree = ElementTree.parse(os.path.join(reports_path, report))
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
            self._write_dataset(dataset_name, images_captions, images_major_tags, images_auto_tags=images_auto_tags)
            self._split_dataset(reports_with_images, dataset_name, split_rate, text_of_reports)

    def download_peir_gross(self, split_rate=0.9):
        """Downloads the peir_gross dataset
        :param split_rate: Percentage of the dataset to be kept as training.
        :return: Writes a folder with the dataset name, in the current working
        directory with the dataset.
        """
        dataset_name = 'peir_gross'
        self._create_directory_for_dataset(dataset_name)
        image_captions, image_tags = self._download_dataset(dataset_name)
        self._write_dataset(dataset_name, image_captions, image_tags)
        self._split_dataset(image_captions, dataset_name, split_rate)

    def _create_directory_for_dataset(self, dataset_folder_name):
        """ Creates directory for a dataset if it does not exist.
        :param dataset_folder_name: Creates a folder with name
        'dataset_folder_name' in the current working directory.
        """
        try:
            # Create target Directory
            path = abspath(path_join(self.dataset_path, dataset_folder_name, dataset_folder_name + "_images"))
            os.makedirs(path)
            self._logger.info("Directory {0} has been created.".format(dataset_folder_name))
            print("Directory ", dataset_folder_name, " Created ")
        except FileExistsError:
            self._logger.info("Directory {0} already exists".format(dataset_folder_name))
            print("Directory ", dataset_folder_name, " already exists")

    def _split_images(self, reports_images, img_keys, filename, text_of_reports=None):
        new_images = {}
        if text_of_reports is None:
            for key in img_keys:
                new_images[key] = reports_images[key]
        else:
            for key in img_keys:
                for img in reports_images[key]:
                    new_images[img] = text_of_reports[key]
        with open(filename, "w") as news_images_file:
            for new_image in new_images:
                news_images_file.write(new_image + "\t" + new_images[new_image])
                news_images_file.write("\n")

    def _write_dataset(self, dataset_folder_name, images_captions, images_tags, images_auto_tags=None):
        """ Writes the dataset in the currect working directory in a folder
        with name ''dataset_folder_name'
        :param dataset: dataset_folder_name
        :param images_captions: list with imgage captions
        :param images_tags: list with images_tags
        :param images_auto_tags: list with auto_tags
        :return:
        """
        self._logger.info("Began writing dataset to file {0}".format(dataset_folder_name))
        write_path = abspath(path_join(self.dataset_path, dataset_folder_name))
        try:
            with open(abspath(path_join(write_path, dataset_folder_name + '.tsv')), "w") as output_file:
                for image_caption in images_captions:
                    output_file.write(image_caption + "\t" + images_captions[image_caption])
                    output_file.write("\n")

            # Safer JSON storing
            with open(abspath(path_join(write_path, dataset_folder_name + '_captions.json')), "w") as output_file:
                output_file.write(json.dumps(images_captions))
            with open(dataset_folder_name + "/" + dataset_folder_name + ".json", "w") as output_file:
                output_file.write(json.dumps(images_tags))
            if images_auto_tags is not None:
                with open(abspath(path_join(write_path, dataset_folder_name + '_auto_tags.json')), "w") as output_file:
                    output_file.write(json.dumps(images_auto_tags))
        except:
            self._logger.exception('Runtime Error/Exception in DownloadDatasets._write_dataset().')

    def _split_dataset(self, data, dataset_name_folder, split_rate, text_of_reports=None):
        """Splits the dataset into training and set.
        :param data: list of data.
        :param dataset_name_folder: name of the folder where the dataset is stored.
        :param split_rate: float number in (0,1). Defines the rate of data to be kept
        for training.
        :param text_of_reports: list with text of reports if they exist.
        :return:
        """
        # perform a case based split
        random.seed(42)
        keys = list(data.keys())
        random.shuffle(keys)

        train_split = int(np.floor(len(data) * split_rate))

        train_keys = keys[:train_split]
        test_keys = keys[train_split:]
        abspath(path_join(self.dataset_path, dataset_name_folder, 'train_images.tsv'))
        train_path = abspath(path_join(self.dataset_path, dataset_name_folder, 'train_images.tsv'))
        test_path = abspath(path_join(self.dataset_path, dataset_name_folder, 'test_images.tsv'))

        self._split_images(data, train_keys, train_path, text_of_reports=text_of_reports)
        self._split_images(data, test_keys, test_path, text_of_reports=text_of_reports)

    def _download_dataset(self, dataset):
        self._logger.info("Downloading {0}".format(dataset))
        if dataset == 'iu_xray':
            os.system("wget -P iu_xray/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz")
            # data reports
            os.system("wget -P iu_xray/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz")
            # unzip
            os.system("tar -xzf ./iu_xray/NLMCXR_png.tgz -C iu_xray/iu_xray_images/")
            os.system("tar -xzf ./iu_xray/NLMCXR_reports.tgz -C iu_xray/")
        else:
            return self._crawl_peir_gross()

    def _crawl_peir_gross(self):
        """Crawls "http://peir.path.uab.edu/library" and
        created the peir_gross dataset.
        """
        image_captions = {}
        image_tags = {}

        base_url = "http://peir.path.uab.edu/library"

        # the main page of the pathology category that contains the collections of all the sub-categories
        main_page_url = "http://peir.path.uab.edu/library/index.php?/category/2"
        main_page = requests.get(main_page_url)
        web_parser = BeautifulSoup(main_page.content, "html.parser")

        # find the links for each sub-category
        categories = web_parser.find("li", class_="selected").find_all("li")
        categories_urls = [category.find("a").get("href") for category in categories]

        # go to each sub-category and extract images from the Gross sub-collection
        for url in categories_urls:
            i = 1
            image_sum = 0

            category_url = base_url + "/" + url
            category_page = requests.get(category_url)
            category_soup = BeautifulSoup(category_page.content, "html.parser")

            # find the Gross sub-collection, if it exists
            collections_urls = {}
            collections = category_soup.find("li", class_="selected").find_all("li")
            for collection in collections:
                name = collection.find("a").get_text()
                collection_url = collection.find("a").get("href")
                collections_urls[name] = collection_url

            if "Gross" in list(collections_urls.keys()):
                # the page of Gross sub-collection to start extracting images from
                page_url = base_url + "/" + collections_urls["Gross"]

                page = requests.get(page_url)
                page_parser = BeautifulSoup(page.content, "html.parser")

                # the url of the last page or empty if there is only one page
                last_page = page_parser.find("a", rel="last")
                if last_page is None:
                    last_page_url = ""
                else:
                    last_page_url = base_url + "/" + last_page.get("href")

                # get the images from all the pages
                while True:
                    # find the links to the images of the current page
                    thumbnails = page_parser.find("ul", class_="thumbnails").find_all("a")

                    for thumbnail in thumbnails:
                        # get the image url
                        image_url = base_url + "/" + thumbnail.get("href")
                        # go to the image page and extract the data
                        image_page = requests.get(image_url)
                        image_soup = BeautifulSoup(image_page.content, "html.parser")

                        image = image_soup.find("img", id="theMainImage")
                        filename = image.get("alt")
                        image_src = image.get("src")
                        description = image.get("title").replace("\r\n", " ")
                        image_captions[filename] = description

                        tags_container = image_soup.find("div", {"id": "Tags"})
                        tags = [tag.string for tag in tags_container.findChildren("a")]
                        image_tags[filename] = tags

                        # save the image to images folder
                        with open("peir_gross/peir_gross_images/" + filename, "wb") as f:
                            image_file = requests.get(base_url + "/" + image_src)
                            f.write(image_file.content)

                    print("Extracted", len(thumbnails), "image-caption pairs from page", i)
                    image_sum = image_sum + len(thumbnails)
                    i += 1

                    # if the current page is the last page stop
                    if page_url == last_page_url or last_page_url == "":
                        print("This was the last page")
                        break

                    # go to the next page
                    page_url = base_url + "/" + page_parser.find("a", rel="next").get("href")
                    page = requests.get(page_url)
                    page_parser = BeautifulSoup(page.content, "html.parser")

                print("Visited", i - 1, "pages of Gross sub-collection")
                print("Extracted", image_sum, "image-caption pairs from the",
                      category_soup.find("li", class_="selected").find("a").get_text(), "category")
                return image_captions, image_tags
