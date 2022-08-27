# from WasteClassifier.model.network import LeNet, ConvolutionalNetwork

ROOT_PATH = '/home/peprycy/WasteClassifier/Data'
TRANSHET_PATH = f'${ROOT_PATH}/TrashNet'
TRASHNET_RESIZED_PATH = f"{TRANSHET_PATH}/dataset-resized"
ORGANIC_PATH = f"{ROOT_PATH}/organic"
TRASHBOX_PATH = f"${ROOT_PATH}/TrashBox"
SPLIT_IMAGES_PATH = f"{ROOT_PATH}/split_images"
CUSTOM_PHOTOS_PATH = f'{ROOT_PATH}/custom_images'
CUSTOM_PHOTOS_RESIZED_PATH = f'{ROOT_PATH}/custom_images_resized'
PREPROCESSED_IMAGES_PATH = f'{ROOT_PATH}/preprocessed_images'
TEST_PERCENT = 0.3
PHOTO_WIDTH = 512
PHOTO_HEIGHT = 384
project_root_path = '/home/peprycy/WasteClassifier'
epochs = 20
learning_rate = 0.001
optimizer = f'torch.optim.Adam(model.parameters(), lr={learning_rate})'
# optimizer = f'torch.optim.SGD(model.parameters(), lr={learning_rate})'
loss_function = 'torch.nn.CrossEntropyLoss()'
# loss_function = 'torch.nn.HingeEmbeddingLoss()'
# loss_function = 'torch.nn.NLLLoss()' # tylko przy softmaxowej aktywacji
binary_loss_function = 'torch.nn.BCELoss()'
batch_size = 10
model_pickle_path = f'{project_root_path}/WasteClassifier/model/model.pickle'
model_plastic_pickle_path = f'{project_root_path}/WasteClassifier/model/model_plastic.pickle'
model_glass_pickle_path = f'{project_root_path}/WasteClassifier/model/model_glass.pickle'
model_metal_pickle_path = f'{project_root_path}/WasteClassifier/model/model_metal.pickle'
model_organic_pickle_path = f'{project_root_path}/WasteClassifier/model/model_organic.pickle'
model_cardboard_pickle_path = f'{project_root_path}/WasteClassifier/model/model_cardboard.pickle'
split_images_path = f'{project_root_path}/Data/split_images'
custom_photos_path = f'{project_root_path}/Data/custom_images'
resized_custom_photos_path = f'{project_root_path}/Data/custom_images_resized'
classes = ['cardboard', 'glass', 'organic', 'plastic']  # nobody ever died from a little of hardcoding, gonna fix it later
plastic_classes = ['plastic', 'not_plasstic']
metal_classes = ['metal', 'not_metal']
glass_classes = ['glass', 'not_glass']
organic_classes = ['organic', 'not_organic']
cardboard_classes = ['cardboard', 'not_cardboard']
classes_dict = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'organic', 4: 'plastic'}  # same here
grayscale = False
channels = 3
binary_train = True
network = 'ConvolutionalNetwork'
hog_transformation = True
