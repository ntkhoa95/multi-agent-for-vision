import os
import urllib.request


def download_file(url, file_path):
    """Download a file from URL to the specified path."""
    try:
        print(f"Downloading {url} to {file_path}")
        urllib.request.urlretrieve(url, file_path)
        print(f"Successfully downloaded to {file_path}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")


def download_test_data():
    """Download test data for local testing."""
    os.makedirs("tests/data/images", exist_ok=True)
    os.makedirs("tests/data/videos", exist_ok=True)

    # URLs for test data
    urls = {
        "tests/data/images/person.jpg": "https://github.com/ultralytics/assets/raw/main/yolo/zidane.jpg",
        "tests/data/images/street.jpg": "https://github.com/ntkhoa95/multi-agent-for-vision/blob/main/tests/data/images/street.jpg",
        "tests/data/images/dog.jpg": "https://github.com/ntkhoa95/multi-agent-for-vision/blob/main/tests/data/images/dog.jpg",
        "tests/data/videos/crosswalk.avi": "https://github.com/ntkhoa95/multi-agent-for-vision/blob/main/tests/data/videos/crosswalk.avi",
    }

    # Download each file
    for file_path, url in urls.items():
        if not os.path.exists(file_path):
            download_file(url, file_path)
        else:
            print(f"File already exists: {file_path}")


if __name__ == "__main__":
    download_test_data()
