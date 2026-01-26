# script for preparing computed embeddings of the ImageNet-1k validation set
# creates a single dictionary that contains the embeddings of all ImageNet-1k classes 

from pathlib import Path 
import pickle 

with open('../resources/vit_s_embeddings/inet_1k_val_cls_pt_1st_half.pkl', 'rb') as embeds_1st_half_pkl: 
    embeds_1st_half = pickle.load(embeds_1st_half_pkl)
    with open('../resources/vit_s_embeddings/inet_1k_val_cls_pt_2nd_half.pkl', 'rb') as embeds_2nd_half_pkl: 
        embeds_2nd_half = pickle.load(embeds_2nd_half_pkl)
        with open('../resources/vit_s_embeddings/inet_1k_val_cls_pt.pkl', 'wb') as embeds_complete_pkl: 
            inet_1k_val_embeds_complete = embeds_1st_half | embeds_2nd_half 
            pickle.dump(inet_1k_val_embeds_complete, embeds_complete_pkl, pickle.HIGHEST_PROTOCOL)


