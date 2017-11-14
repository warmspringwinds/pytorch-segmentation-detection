import torch.utils.data as data

class Endovis_Instrument_Unified(data.Dataset):
    
    def __init__(self,
                 endovis_2017_dataset_obj,
                 endovis_2015_dataset_obj):
        
        self.endovis_2017_dataset_obj = endovis_2017_dataset_obj
        self.endovis_2015_dataset_obj = endovis_2015_dataset_obj
        
        self.endovis_2017_dataset_size = len(endovis_2017_dataset_obj)
        self.endovis_2015_dataset_size = len(endovis_2015_dataset_obj)
        
    def __len__(self):
        
        return self.endovis_2017_dataset_size + self.endovis_2015_dataset_size
    
    def __getitem__(self, index):
        
        if index >= self.endovis_2017_dataset_size:
            
            index -= self.endovis_2017_dataset_size
            
            return self.endovis_2015_dataset_obj[index]
        
        return self.endovis_2017_dataset_obj[index]