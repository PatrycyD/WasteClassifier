import cv2
import config
import numpy as np
from WasteClassifier.model.network import ConvolutionalNetwork
import torch
from torchvision import transforms
from torch.autograd import Variable


model = ConvolutionalNetwork(len(config.classes))
model.load_pickle(config.model_pickle_path)


class LiveDetector:

    def __init__(self, net):

        self.image_x = config.PHOTO_WIDTH
        self.image_y = config.PHOTO_HEIGHT

        self.class_labels = config.classes
        self.class_labels.sort()

        self.label_count = len(self.class_labels)

        torch.manual_seed(1)
        self.net = net

    def forward(self, input_image):

        self.net.eval()

        input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)
        input_image_wrapped = Variable(input_image)

        feed_forward_results = self.net(input_image_wrapped)
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return class_label.data[0]

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
        normalizer = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        transforms.CenterCrop(224),
                                        ])

        print("Starting Real Time Classification Video Stream. Hit 'Q' to Exit.")
        while True:
            is_read, read_frame = video_capture.read()

            if is_read:
                read_frame_scaled = cv2.resize(read_frame, (config.PHOTO_WIDTH, config.PHOTO_HEIGHT),
                                               interpolation=cv2.INTER_LINEAR)
                read_frame_normalized = normalizer(read_frame_scaled)

                class_label_index = self.forward(read_frame_normalized)
                class_label_string = self.class_labels[class_label_index]

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
    live_predictor = LiveDetector(model)
    live_predictor.cam(0)
