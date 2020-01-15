import os
import json
import random
import requests
import numpy as np
from os.path import join as path_join, abspath
from bs4 import BeautifulSoup

PATH = os.getcwd()


def create_directory_for_dataset(dataset_folder_name):
    """ Creates directory for a dataset if it does not exist.
    :param dataset_folder_name: Creates a folder with name
    'dataset_folder_name' in the current working directory.
    """
    try:
        # Create target Directory
        path = abspath(path_join(PATH, dataset_folder_name, dataset_folder_name+"_images"))
        os.makedirs(path)
        print("Directory ", dataset_folder_name, " Created ")
    except FileExistsError:
        print("Directory ", dataset_folder_name, " already exists")


def split_images(reports_images, img_keys, filename, text_of_reports=None):
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


def write_dataset(dataset_folder_name, images_captions, images_tags, images_auto_tags=None):
    """ Writes the dataset in the currect working directory in a folder
    with name ''dataset_folder_name'
    :param dataset: dataset_folder_name
    :param images_captions: list with imgage captions
    :param images_tags: list with images_tags
    :param images_auto_tags: list with auto_tags
    :return:
    """
    write_path = abspath(path_join(PATH, dataset_folder_name))
    with open(abspath(path_join(write_path, dataset_folder_name+'.tsv')), "w") as output_file:
        for image_caption in images_captions:
            output_file.write(image_caption + "\t" + images_captions[image_caption])
            output_file.write("\n")

    # Safer JSON storing
    with open(abspath(path_join(write_path, dataset_folder_name+'_captions.json')), "w") as output_file:
        output_file.write(json.dumps(images_captions))
    with open(dataset_folder_name + "/" + dataset_folder_name + ".json", "w") as output_file:
        output_file.write(json.dumps(images_tags))
    if images_auto_tags is not None:
        with open(abspath(path_join(write_path, dataset_folder_name+'_auto_tags.jsonn')), "w") as output_file:
            output_file.write(json.dumps(images_auto_tags))


def split_dataset(data, dataset_name_folder, split_rate, text_of_reports=None):
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
    abspath(path_join(PATH, dataset_name_folder,'train_images.tsv'))
    train_path = abspath(path_join(PATH, dataset_name_folder, 'train_images.tsv'))
    test_path = abspath(path_join(PATH, dataset_name_folder, 'test_images.tsv'))

    split_images(data, train_keys, train_path, text_of_reports=text_of_reports)
    split_images(data, test_keys, test_path, text_of_reports=text_of_reports)


def download_dataset(dataset):
    if dataset is 'iu_xray':
        os.system("wget -P iu_xray/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz")
        # download reports
        os.system("wget -P iu_xray/ https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz")
        # unzip
        os.system("tar -xzf ./iu_xray/NLMCXR_png.tgz -C iu_xray/iu_xray_images/")
        os.system("tar -xzf ./iu_xray/NLMCXR_reports.tgz -C iu_xray/")
    elif dataset is 'peir_gross':
        return crawl_peir_gross()
    else:
        pass


def crawl_peir_gross():
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
                page_url = base_url + "/" + page_soup.find("a", rel="next").get("href")
                page = requests.get(page_url)
                page_soup = BeautifulSoup(page.content, "html.parser")

            print("Visited", i - 1, "pages of Gross sub-collection")
            print("Extracted", image_sum, "image-caption pairs from the",
                  category_soup.find("li", class_="selected").find("a").get_text(), "category")
            return image_captions, image_tags
