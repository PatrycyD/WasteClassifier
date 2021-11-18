import pathlib

ROOT_PATH = pathlib.Path('c:\\', 'Users', 'partycy', 'PycharmProjects', 'WasteClassifier', 'Data', 'TrashNet')
# ROOT_PATH = 'C:\Users\patrycy\PycharmProjects\WasteClassifier\Data\TrashNet'

RESIZED_PATH = pathlib.Path(ROOT_PATH, 'dataset-resized')
# RESIZED_PATH = f'{ROOT_PATH}\dataset-resized'

SPLITTED_IMAGES_PATH = pathlib.Path(ROOT_PATH, 'splitted_images')
# SPLITTED_IMAGES_PATH = f'{ROOT_PATH}\splitted_images'

TEST_PERCENT = 0.3
