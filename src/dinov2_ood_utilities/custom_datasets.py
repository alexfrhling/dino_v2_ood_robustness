from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Union  
from pathlib import Path
import numpy as np
import torch
import csv

# Dataset where classes are defined by directories + exclude directories from loading by instantiation 

class CustomizedImageFolder(ImageFolder): 

    def __init__(self, not_processed_imagenet_classes = None, root = None, transform = None):
        self.not_processed_imagenet_classes = not_processed_imagenet_classes 
        super().__init__(root = root, transform = transform)

    def find_classes(self, directory: Union[str, Path]) -> tuple[list[str], dict[str, int]]:

        class_list = []
        class_index_dict = dict()
    
        with open('../resources/imagenet_train_class_to_index_mapping.csv', 'r') as class_index_table:

            class_index_reader = csv.reader(class_index_table, delimiter=';')
            for table_row in class_index_reader:
                class_wnid = table_row[0]
                class_index = table_row[1]

                if class_wnid in self.not_processed_imagenet_classes:
                    class_list.append(class_wnid)
                    class_index_dict[class_wnid] = class_index

        return class_list, class_index_dict
    
# Dataset where classes are defined by directories + exclude directories from loading by instantiation 

class CustomizedImageFolderForImagenetV2(ImageFolder): 

    def __init__(self, not_processed_imagenet_classes = None, root = None, transform = None):
        self.not_processed_imagenet_classes = not_processed_imagenet_classes 
        super().__init__(root = root, transform = transform)

    def find_classes(self, directory: Union[str, Path]) -> tuple[list[str], dict[str, int]]:

        class_list = []
        class_index_dict = dict()
    
        with open('../resources/imagenet_train_class_to_index_mapping.csv', 'r') as class_index_table:
            with open('../resources/imagenet_1k_label_order.txt', 'r') as label_order_file:
                inet_1k_labels = label_order_file.readlines()
                inet_1k_labels = [label_order_line.split()[0] for label_order_line in inet_1k_labels]

                class_index_reader = csv.reader(class_index_table, delimiter=';')
                for table_row in class_index_reader:
                    class_wnid = table_row[0]
                    class_index = table_row[1]

                    inet_1k_label_pos = inet_1k_labels.index(class_wnid)
                    if class_wnid in self.not_processed_imagenet_classes:
                        class_list.append(class_wnid)
                        class_index_dict[str(inet_1k_label_pos)] = class_index

        return class_list, class_index_dict
    

# DataSet for training LinearClassifier
# custom dataset: dictonary -> [(embedding, index_as_tensor)]

class DictionaryDataset(torch.utils.data.Dataset): 
    
    """
    Pararms:
    1) data: complete dataset provided as dict
    2) index_list: list of data_dict-keys, order of keys in list will be used to create a tensor as the expected model-output
    """

    def __init__(self, data: dict, index_list: list[str]): 
        self.data_dict = data
        self.index_list = index_list 
        self.wnid_list = list(self.data_dict.keys())
        self.wnid_iterator = iter(self.wnid_list)
        self.instance_per_wnid = [len(self.data_dict[key]) for key in self.wnid_list]

    def __len__(self) -> int: 
        total_len = 0 
        for key in self.wnid_list:
            total_len += len(self.data_dict[key])

        return total_len 
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]: 

        sum_instances = 0
        sum_rest = 0

        # start with first wnid in list
        new_wnid = ''

        # iterate until total of all already covered instances is bigger than index of interest
        # when class of interest is reached, condition will be false (sum_instances points to first element of next class)
        for wnid in self.wnid_list: 
            if sum_instances > idx:
                break 
            new_wnid = wnid 
            sum_rest = sum_instances 
            sum_instances += len(self.data_dict[wnid])

        # index within class is needed 
        # fixed order: [class1, class2, ...]
        # when lenght of each previously covered class is known, index within class of interest can be calculated 
        # sum_rest always represent the total of instances of all covered classes yet, not of interest
        idx_within_class = idx - sum_rest

        np_arr = self.data_dict[new_wnid][idx_within_class]
        data_tensor = torch.tensor(np.array(np_arr))

        label_tensor = torch.zeros(len(self.index_list))
        # put a one at the position for the expected class 
        label_tensor[self.index_list.index(new_wnid)] = 1 

        return data_tensor, label_tensor
    