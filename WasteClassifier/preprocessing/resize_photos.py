import cv2
import WasteClassifier.config as config
import shutil
import pathlib


def main(source_path: str, target_path: str, depth: int):
    target_path = pathlib.Path(target_path)
    source_path = pathlib.Path(source_path)

    if target_path.is_dir() and len([x for x in target_path.iterdir()]) != 0:
        shutil.rmtree(target_path)
    elif target_path.is_dir():
        pathlib.Path(target_path).rmdir()

    target_path.mkdir()

    if depth == 0:
        count = 0

        for file_name in source_path.iterdir():
            source_file_path = f'{source_path}/{file_name}'
            target_file_path = f'{target_path}/{file_name}'
            resize_file(source_file_path, target_file_path)
            count += 1

    # I know it hurts, but this is just python file manipulation stuff
    elif depth == 1:
        count = 0

        for label_path in source_path.iterdir():

            for file_name in pathlib.Path(label_path).iterdir():
                target_file_path = pathlib.Path(target_path) / file_name.name
                resize_file(file_name, target_file_path)
                count += 1


def resize_file(source_file_path, destination_file_path):
    img = cv2.imread(str(source_file_path))
    resized = cv2.resize(img, (config.PHOTO_WIDTH, config.PHOTO_HEIGHT))
    cv2.imwrite(str(destination_file_path), resized)


if __name__ == '__main__':
    main(config.CUSTOM_PHOTOS_PATH, config.CUSTOM_PHOTOS_RESIZED_PATH, 1)
