
from torchvision import transforms
from pathlib import Path
import torch 
import timm
import glob
import csv
import pickle

from dinov2_ood_utilities.custom_datasets import CustomizedImageFolder, CustomizedImageFolderForImagenetV2 


INET_C_EMBEDS_STORE_PATH = '../resources/vit_s_embeddings/imagenet_c'
#INET_C_SRC_PATH = '../datasets/ImageNetC'
INET_C_SRC_PATH = '/home/stud/afroehli/datasets/ImagenetC'

# load general list for label to wnid mapping
class_to_index_mapping = []
with open('../resources/imagenet_train_class_to_index_mapping.csv', 'r') as class_index_table:
    class_index_reader = csv.reader(class_index_table, delimiter=';')
    for inet_class, _ in class_index_reader: 
        class_to_index_mapping.append(inet_class)

# define input image transformations
timm_model = 'vit_small_patch14_dinov2'
timm_model_conf = timm.data.resolve_model_data_config(timm_model)
timm_model_conf['input_size'] = (3, 518, 518)
timm_transform = timm.data.create_transform(**timm_model_conf, is_training=False)
# print(f'Following transform will be applied: {timm_transform}')

# load list with all ImageNet-1k wnids
with open('../resources/imagenet_1k_label_order.txt', 'r') as label_order_file:
    inet_1k_labels = label_order_file.readlines()
    inet_1k_labels = [label_order_line.split()[0] for label_order_line in inet_1k_labels]


# create list of dataloaders, one for each of the 95 ImageNet-C datasets
inet_c_src_paths = glob.glob(f'{INET_C_SRC_PATH}/*/*')
dataloaders = []
for dset_path in inet_c_src_paths:
    dset = CustomizedImageFolder(not_processed_imagenet_classes=inet_1k_labels, root=dset_path,
                                                transform=timm_transform)
    cor_type = dset_path.split('/')[-2:]
    dataloaders.append((torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True), 
                       f'{INET_C_EMBEDS_STORE_PATH}/{cor_type[0]}/sev_{cor_type[1]}.pkl',
                       f'{INET_C_EMBEDS_STORE_PATH}/{cor_type[0]}'))
# print(f'Number of created dataloaders: {len(dataloaders)}')
    

# define model to be used 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {device}')
vision_transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
vision_transformer.eval()
vision_transformer.to(device)


# compute ImageNet-C embeddings 

N_LAYERS = 1

with torch.no_grad():

    # debug info
    with open('./compute_inet_c_embeds_progress.txt', 'w') as prog_file:
        prog_file.write('# Begin computation of ImageNet-C embeddings\n')

    for loader_n, (loader, str_path, str_path_dir) in enumerate(dataloaders):
        
        cor_type_loader = str_path.split('/')[-2:]
        # print(f'Next calculate results for dataset: {f'{cor_type_loader[0]}_{cor_type_loader[1]}'.removesuffix('.pkl')}')
        # print(f'Will be stored under: {str_path}')

        batches_per_loader = len(loader)
        history_processed_batches = 0

        # create new results-dict for each loader 
        model_results = dict()

        for batch_n, (samples, sample_labels) in enumerate(loader):

            single_samples = torch.unbind(samples, dim=0)
            samples_out = []
            for single_sample in single_samples:
                single_sample_out = vision_transformer.get_intermediate_layers(single_sample.unsqueeze(0).to(device), N_LAYERS, return_class_token=True)
                samples_out.append((torch.mean(single_sample_out[0][0], dim=1).cpu().detach().numpy(), single_sample_out[0][1].cpu().detach().numpy()))

            wnid_per_sample = [class_to_index_mapping[int(sample_label)] for sample_label in sample_labels]

            for n, sample_out in enumerate(samples_out):
                sample_item_wnid = wnid_per_sample[n]
                try: 
                    model_results[sample_item_wnid].append(sample_out)
                except KeyError:
                    model_results[sample_item_wnid] = [sample_out]
                    
        try:
            with open(str_path, 'wb') as pkl_file:
                pickle.dump(model_results, pkl_file, pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            Path(str_path_dir).mkdir(parents=True, exist_ok=True)
            with open(str_path, 'wb') as pkl_file:
                pickle.dump(model_results, pkl_file, pickle.HIGHEST_PROTOCOL)

        # debug info
        with open('./compute_inet_c_embeds_progress.txt', 'a') as prog_file:
            prog = ((loader_n + 1) / len(dataloaders)) * 100 
            prog_file.write(f'N-Loader: {loader_n}, Progress: {prog}%, Completed: {f'{cor_type_loader[0]}_{cor_type_loader[1]}'.removesuffix('.pkl')}\n')
# debug info
with open('./compute_inet_c_embeds_progress.txt', 'a') as prog_file:
        prog_file.write('# Terminate computation of ImageNet-C embeddings\n')