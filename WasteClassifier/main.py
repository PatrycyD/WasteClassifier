import cv2
import config
import numpy as np
from WasteClassifier.model.network import ConvolutionalNetwork, HOGNeuralNetwork, get_inception_nn
import torch
from torchvision import transforms
from torch.autograd import Variable


class LiveDetector:

    def __init__(self, binary):
        self.binary = binary
        self.image_x = config.PHOTO_WIDTH
        self.image_y = config.PHOTO_HEIGHT

        if not binary:
            self.class_labels = config.classes
            self.class_labels.sort()
            self.label_count = len(self.class_labels)

        self.net = None
        self.nets = None
        self.model_plastic = None
        self.model_glass = None
        self.model_cardboard = None
        self.model_organic = None

    def load_models(self, model_type, pickle_path):

        if hog_transform:
            pickle_path = pickle_path.replace('.pickle', '_hog.pickle')

        torch.manual_seed(1)

        if self.binary:
            self.model_cardboard = model
            if callable(getattr(self.model_cardboard, 'load_pickle', None)):
                self.model_cardboard.load_pickle('model_cardboard'.join(pickle_path.rsplit('model', 1)))  # replace last occurence

            self.model_glass = model
            if callable(getattr(self.model_glass, 'load_pickle', None)):
                self.model_glass.load_pickle('model_glass'.join(pickle_path.rsplit('model', 1)))

            # self.model_metal = ConvolutionalNetwork(1, config.channels)
            # self.model_metal.load_pickle(config.model_metal_pickle_path)
            # self.model_metal.eval()

            self.model_organic = model
            if callable(getattr(self.model_organic, 'load_pickle', None)):
                self.model_organic.load_pickle('model_organic'.join(pickle_path.rsplit('model', 1)))

            self.model_plastic = model
            if callable(getattr(self.model_plastic, 'load_pickle', None)):
                self.model_plastic.load_pickle('model_plastic'.join(pickle_path.rsplit('model', 1)))

            self.nets = [self.model_cardboard, self.model_glass, #self.model_metal,
                         self.model_organic, self.model_plastic]

        else:
            self.net = model
            if callable(getattr(self.net, 'load_pickle', None)):
                self.net.load_pickle(pickle_path)

    def forward(self, input_image):
        self.net.eval()

        input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)
        input_image_wrapped = Variable(input_image)

        feed_forward_results = self.net(input_image_wrapped)
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return prob, class_label.data[0]

    def forward_binary(self, input_image):

        input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)
        input_image_wrapped = Variable(input_image)
        print('a')
        print(self.model_glass.forward(input_image_wrapped))
        print(self.model_cardboard.forward(input_image_wrapped))
        print(self.model_organic.forward(input_image_wrapped))
        print(self.model_plastic.forward(input_image_wrapped))
        probs = [net.forward(input_image_wrapped).item() for net in self.nets]
        predicted_class = config.classes[probs.index(max(probs))]
        probability = max(probs)

        return probability, predicted_class

    def view(self, image_to_show):
        class_label_index = self.forward(image_to_show)
        class_label_string = self.class_labels[class_label_index]

        image_to_show = image_to_show.type(torch.FloatTensor)
        image_to_show = np.transpose(image_to_show.numpy(), (1, 2, 0))

        cv2.namedWindow(class_label_string, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(class_label_string, 800, 500)
        cv2.imshow(class_label_string, image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cam(self, idx=0):
        video_capture = cv2.VideoCapture(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_location = (50, 450)
        font_scale = 3
        font_color = (255, 255, 255)
        line_type = 2
        video_capture.set(3, 720)
        video_capture.set(4, 720)
        grayscale = config.grayscale
        if grayscale:
            norm_1st = 0.485
            norm_2nd = 0.229
        else:
            norm_1st = [0.485, 0.456, 0.406]
            norm_2nd = [0.229, 0.224, 0.225]

        transforms_list = [
                           transforms.ToTensor(),
                           transforms.CenterCrop((384, 512)),
                           transforms.Normalize(norm_1st, norm_2nd)
                           ]

        if grayscale:
            transforms_list.append(transforms.Grayscale())

        transform = transforms.Compose(transforms_list)

        print("Starting Real Time Classification Video Stream. Hit 'Q' to Exit.")
        while True:
            is_read, read_frame = video_capture.read()
            # if config.is_hsv:
            #     read_frame_hsv = cv2.cvtColor(read_frame, cv2.COLOR_BGR2HSV)
            if is_read:
                read_frame_normalized = transform(read_frame)

                with torch.no_grad():
                    if self.binary:
                        probability, class_name = self.forward_binary(read_frame_normalized)

                    else:
                        probability, class_label_index = self.forward(read_frame_normalized)
                        class_name = self.class_labels[class_label_index]

                if isinstance(probability, torch.Tensor):
                    probability = probability.item()

                class_label_string = class_name
                # class_label_string = class_name + '-' + str(round(probability, 3) * 100) + '%'
                cv2.putText(read_frame, class_label_string, text_location, font, font_scale, font_color, line_type)
                cv2.imshow('Real Time Classification Video Stream', read_frame)
                cv2.resizeWindow('Real Time Classification Video Stream', 500, 300)

            else:
                print('I/O Error whilst capturing video feed.')
                break

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    use_binaries = config.binary_train
    hog_transform = config.hog_transformation
    if not use_binaries:
        num_classes = len(config.classes)
    else:
        num_classes = 2 if config.architecture == 'inception' else 1

    if hog_transform:
        model = HOGNeuralNetwork(num_classes, config.hog_nn_input_params)
    elif config.architecture == 'inception':
        # model = get_inception_nn(num_classes, False, config.model_pickle_path)
        model = torch.load('/home/peprycy/WasteClassifier/WasteClassifier/model/model_full_model.pickle')
    else:
        model = ConvolutionalNetwork(num_classes, config.channels)

    live_predictor = LiveDetector(use_binaries)
    live_predictor.load_models(model, config.model_pickle_path)
    live_predictor.cam(0)
