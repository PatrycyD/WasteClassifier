from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.functional as F
import os
import psutil
from WasteClassifier.model.network import ConvolutionalNetwork
from WasteClassifier import config


model = ConvolutionalNetwork(len(config.classes))
model.load_pickle(config.model_pickle_path)


class Img2Obj:

    def __init__(self, model):

        learning_rate = 0.001

        #Images are 3 x 32 x 32
        self.image_x = config.PHOTO_WIDTH
        self.image_y = config.PHOTO_HEIGHT

        #CIFAR 100 Dataset
        self.class_labels = config.classes
        self.class_labels.sort()

        self.label_count = len(self.class_labels)

        torch.manual_seed(1)
        self.model = model

    def forward(self, input_image): #will be 28 x 28 ByteTensor
        #print "In NnImg2Num Forward"
        self.model.eval()
        #input_image_wrapped = Variable(torch.unsqueeze(input_image, 0))
        input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)
        input_image_wrapped = Variable(input_image)

        feed_forward_results = self.model(input_image_wrapped)
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return class_label.data[0]

    def view(self, image_to_show):
        class_label_index = self.forward(image_to_show)
        class_label_string = self.class_labels[class_label_index]
        # print class_label
        # print self.class_labels
        # print self.class_labels[class_label]
        #class_label = "Class Label: "+str(self.class_labels[class_label])

        image_to_show = image_to_show.type(torch.FloatTensor)
        image_to_show = np.transpose(image_to_show.numpy(), (1, 2, 0))

        cv2.namedWindow(class_label_string, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(class_label_string, 400, 400)
        cv2.imshow(class_label_string, image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def cam(self, idx=0):
        video_capture = cv2.VideoCapture(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX  #cv2.FONT_HERSHEY_PLAIN
        textLocation = (50, 450)
        fontScale = 3
        fontColor = (255, 255, 255)
        lineType = 2
        video_capture.set(3, 720)
        video_capture.set(4, 720)
        normalizer = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        transforms.CenterCrop(224),
                                        ])

        print("Starting Real Time Classification Video Stream. Hit 'Q' to Exit.")
        while True:
            isRead, read_frame = video_capture.read()

            if isRead:
                read_frame_scaled = cv2.resize(read_frame, (config.PHOTO_WIDTH, config.PHOTO_HEIGHT),
                                               interpolation=cv2.INTER_LINEAR)
                read_frame_normalized = normalizer(read_frame_scaled)

                class_label_index = self.forward(read_frame_normalized)
                class_label_string = self.class_labels[class_label_index]

                cv2.putText(read_frame, class_label_string, textLocation, font, fontScale, fontColor, lineType)
                cv2.imshow('Real Time Classification Video Stream', read_frame)

            else:
                print('I/O Error whilst capturing video feed.')
                break

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

viewer = Img2Obj(model)
viewer.cam(0)