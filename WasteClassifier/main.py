import cv2
import config
import numpy as np
from WasteClassifier.model.network import ConvolutionalNetwork
import torch
from torchvision import transforms
from torch.autograd import Variable
from WasteClassifier.preprocessing.images_preprocessing import convert_img_to_nn_input


class LiveDetector:

    def __init__(self, nets, binary):
        self.binary = binary
        self.image_x = config.PHOTO_WIDTH
        self.image_y = config.PHOTO_HEIGHT

        if not binary:
            self.class_labels = config.classes
            self.class_labels.sort()
            self.label_count = len(self.class_labels)

        torch.manual_seed(1)
        if not binary:
            self.net = nets
        else:
            self.nets = nets
            # fixed length
            self.model_plastic = nets[0]
            self.model_glass = nets[1]
            self.model_cardboard = nets[2]
            self.model_organic = nets[3]
            # self.model_metal = nets[4]

    def forward(self, input_image):

        self.net.eval()

        input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)
        input_image_wrapped = Variable(input_image)

        feed_forward_results = self.net(input_image_wrapped)
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return prob, class_label.data[0]

    def forward_binary(self, input_image):
        # for net in self.nets:
        #     net.eval()

        input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)
        input_image_wrapped = Variable(input_image)

        probs = [net.forward(input_image_wrapped).item() for net in self.nets]
        predicted_class = binary_order[probs.index(max(probs))]
        probability = max(probs)

        return probability, predicted_class

    def view(self, image_to_show):
        class_label_index = self.forward(image_to_show)
        class_label_string = self.class_labels[class_label_index]

        image_to_show = image_to_show.type(torch.FloatTensor)
        image_to_show = np.transpose(image_to_show.numpy(), (1, 2, 0))

        cv2.namedWindow(class_label_string, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(class_label_string, 400, 400)
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

            if is_read:
                # read_frame_scaled = cv2.resize(read_frame, (config.PHOTO_WIDTH, config.PHOTO_HEIGHT),
                #                                interpolation=cv2.INTER_LINEAR)
                # read_frame_transformed = convert_img_to_nn_input(read_frame, True)
                # read_frame_normalized = transform(read_frame_transformed)
                read_frame_normalized = transform(read_frame)
                # print(read_frame_normalized.shape)

                if self.binary:
                    probability, class_name = self.forward_binary(read_frame_normalized)
                else:
                    probability, class_label_index = self.forward(read_frame_normalized)
                    class_name = self.class_labels[class_label_index]

                # class_label_string = class_name + '-' + str(round(probability.item(), 3)*100) + '%'
                class_label_string = class_name + '-' + str(round(probability, 3) * 100) + '%'
                cv2.putText(read_frame, class_label_string, text_location, font, font_scale, font_color, line_type)
                cv2.imshow('Real Time Classification Video Stream', read_frame)

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
    if not use_binaries:
        model = ConvolutionalNetwork(len(config.classes), config.channels)
        model.load_pickle(config.model_pickle_path)
        live_predictor = LiveDetector(model, use_binaries)
        live_predictor.cam(0)
    else:
        binary_order = ['plastic', 'glass', 'cardboard', 'organic', 'metal']
        model_plastic = ConvolutionalNetwork(1, config.channels)
        model_plastic.load_pickle(config.model_plastic_pickle_path)

        model_glass = ConvolutionalNetwork(1, config.channels)
        model_glass.load_pickle(config.model_glass_pickle_path)

        model_cardboard = ConvolutionalNetwork(1, config.channels)
        model_cardboard.load_pickle(config.model_cardboard_pickle_path)

        model_organic = ConvolutionalNetwork(1, config.channels)
        model_organic.load_pickle(config.model_organic_pickle_path)

        # model_metal = ConvolutionalNetwork(1, config.channels)
        # model_metal.load_pickle(config.model_metal_pickle_path)

        all_models = [model_plastic, model_glass, model_cardboard, model_organic]#, model_metal]
        live_predictor = LiveDetector(all_models, use_binaries)
        live_predictor.cam(0)
