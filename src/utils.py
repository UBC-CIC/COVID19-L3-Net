import torch

def collate_fn(batch):
    batch_dict = {}
    for k in batch[0]:
        batch_dict[k] = []
        for i in range(len(batch)):
            
            batch_dict[k] += [batch[i][k]]
    # tuple(zip(*batch))
    batch_dict['images'] = torch.stack(batch_dict['images'])
    batch_dict['masks'] = torch.stack(batch_dict['masks'])

    return batch_dict 
    