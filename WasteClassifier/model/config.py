project_root_path = '/home/peprycy/WasteClassifier'
epochs = 5
optimizer = 'torch.optim.Adam(model.parameters(), lr=0.001)'
loss_function = 'torch.nn.CrossEntropyLoss()'
batch_size = 10
model_pickle_path = f'{project_root_path}/WasteClassifier/model/model.pickle'
trashnet_path = f'{project_root_path}/Data/TrashNet/split_images'
custom_photos_path = f'{project_root_path}/Data/custom_images'
