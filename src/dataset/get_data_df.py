from typing import Tuple, Union


from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchio as tio
from pathlib import Path
import pandas as pd
import torch
import os
# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    # tio.Resize(target_shape=(128, 128, 128))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

class BREASTSpadeDatasetDF(Dataset):
    def __init__(self, df: str, phase: str = "train"):
        super().__init__()
        self.df_dir = df
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        # self.train_ds, self.valid_ds = self.get_data_files_df()
        self.train_ds = self.get_train_df() if phase == "train" else None
        self.valid_ds = self.get_train_df() if phase == "valid" else None
        if phase == "train":
            self.dataset = self.train_ds
        elif phase == "valid":
            self.dataset = self.valid_ds
        else:
            raise ValueError("phase should be either 'train' or 'valid'")
    def get_train_df(self):
        train_df = pd.read_csv(self.df_dir)

        essential_fields = ["patient_id","image_id", "image_path", "pcr", "age", "tumor_subtype", "her2", "seg_path"]

        # Filter data to include only essential fields
        filtered_train_ds = [{key: item[key] for key in essential_fields} for item in train_df.to_dict("records")]
        return filtered_train_ds

    def get_data_files_df(self):
        df = pd.read_csv(self.df_dir)
        # Create train, validation and test dataset.
        df['age'] = pd.to_numeric(df['age'].fillna(df['age'].median()), errors='coerce')
        df['her2'] = pd.to_numeric(df['her2'].fillna(df['her2'].median()), errors='coerce')
        df['tumor_subtype'] = df['tumor_subtype'].fillna('Unknown')

        unique_df = df.drop_duplicates(subset='patient_id', keep='first')

        unique_train_df, unique_validation_df = train_test_split(
                unique_df,
                train_size=0.80,
                test_size=0.20,
                stratify=unique_df['pcr'],
                random_state=0,
                shuffle=True,)
        # Filter the original DataFrame based on train_df patient_id and keep the first two image_id entries
        train_df = df[df['patient_id'].isin(unique_train_df['patient_id'])]
        # train_df = train_df.sort_values(by=['patient_id', 'image_id'])
        train_df = train_df.sort_values(by=['patient_id'])
        # train_df = train_df.groupby('patient_id').head(images_per_patient).reset_index(drop=True)

        validation_df = df[df['patient_id'].isin(unique_validation_df['patient_id'])]
        # validation_df = validation_df.sort_values(by=['patient_id', 'image_id'])
        validation_df = validation_df.sort_values(by=['patient_id'])
        # validation_df = validation_df.groupby('patient_id').head(images_per_patient).reset_index(drop=True) # taking the first two aquisitions of each patient

        #  Define the essential fields
        essential_fields = ["patient_id","image_id", "image_path", "pcr", "age", "tumor_subtype", "her2", "seg_path"]

        # Filter data to include only essential fields
        filtered_train_ds = [{key: item[key] for key in essential_fields} for item in train_df.to_dict("records")]
        filtered_validation_ds = [{key: item[key] for key in essential_fields} for item in validation_df.to_dict("records")]
        return filtered_train_ds, filtered_validation_ds
    
    def __len__(self):
        return len(self.dataset)
    


    def __getitem__(self, idx: int):

        data = self.dataset[idx]
        assert isinstance(data, dict), "data should be a dictionary"
  

        img = tio.ScalarImage(data['image_path'])
        label = tio.LabelMap(data['seg_path'])
        tio_sub = tio.Subject(image=img, label=label)
        tio_sub = self.preprocessing(tio_sub)
        tio_sub = self.transforms(tio_sub)
        # return {'data': img.data.permute(0, -1, 1, 2)}
        return {'image': tio_sub['image'].data,
                'label': tio_sub['label'].data,
                # 'data': data,
                }
    



class BREASTSpadeDatasetDF_valid(Dataset):
    def __init__(self, df: str, phase: str = "train"):
        super().__init__()
        self.df_dir = df
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.train_ds, self.valid_ds = self.get_data_files_df()
        if phase == "train":
            self.dataset = self.train_ds
        elif phase == "valid":
            self.dataset = self.valid_ds
        else:
            raise ValueError("phase should be either 'train' or 'valid'")


    def get_data_files_df(self):
        df = pd.read_csv(self.df_dir)
        # Create train, validation and test dataset.
        df['age'] = pd.to_numeric(df['age'].fillna(df['age'].median()), errors='coerce')
        df['her2'] = pd.to_numeric(df['her2'].fillna(df['her2'].median()), errors='coerce')
        df['tumor_subtype'] = df['tumor_subtype'].fillna('Unknown')

        unique_df = df.drop_duplicates(subset='patient_id', keep='first')

        unique_train_df, unique_validation_df = train_test_split(
                unique_df,
                train_size=0.80,
                test_size=0.20,
                stratify=unique_df['pcr'],
                random_state=0,
                shuffle=True,)
        # Filter the original DataFrame based on train_df patient_id and keep the first two image_id entries
        train_df = df[df['patient_id'].isin(unique_train_df['patient_id'])]
        # train_df = train_df.sort_values(by=['patient_id', 'image_id'])
        train_df = train_df.sort_values(by=['patient_id'])
        # train_df = train_df.groupby('patient_id').head(images_per_patient).reset_index(drop=True)

        validation_df = df[df['patient_id'].isin(unique_validation_df['patient_id'])]
        # validation_df = validation_df.sort_values(by=['patient_id', 'image_id'])
        validation_df = validation_df.sort_values(by=['patient_id'])
        # validation_df = validation_df.groupby('patient_id').head(images_per_patient).reset_index(drop=True) # taking the first two aquisitions of each patient

        #  Define the essential fields
        essential_fields = ["patient_id","image_id", "image_path", "pcr", "age", "tumor_subtype", "her2", "seg_path"]

        # Filter data to include only essential fields
        filtered_train_ds = [{key: item[key] for key in essential_fields} for item in train_df.to_dict("records")]
        filtered_validation_ds = [{key: item[key] for key in essential_fields} for item in validation_df.to_dict("records")]
        return filtered_train_ds, filtered_validation_ds
    
    def __len__(self):
        return len(self.dataset)
    


    def __getitem__(self, idx: int):

        data = self.dataset[idx]
        assert isinstance(data, dict), "data should be a dictionary"
  

        img = tio.ScalarImage(data['image_path'])
        label = tio.LabelMap(data['seg_path'])
        tio_sub = tio.Subject(image=img, label=label)
        tio_sub = self.preprocessing(tio_sub)
        # tio_sub = self.transforms(tio_sub)
        # return {'data': img.data.permute(0, -1, 1, 2)}
        return {'image': tio_sub['image'].data,
                'label': tio_sub['label'].data,
                'data': data,
                }
    



class BREASTSpadeDatasetDF_generate(Dataset):
    def __init__(self, df: str, phase: str = "train"):
        super().__init__()
        self.df_dir = df
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        # self.train_ds, self.valid_ds = self.get_data_files_df()
        self.train_ds = self.get_train_df()
        if phase == "train":
            self.dataset = self.train_ds
        elif phase == "valid":
            self.dataset = self.valid_ds
        else:
            raise ValueError("phase should be either 'train' or 'valid'")


    def get_data_files_df(self):
        df = pd.read_csv(self.df_dir)
        # Create train, validation and test dataset.
        df['age'] = pd.to_numeric(df['age'].fillna(df['age'].median()), errors='coerce')
        df['her2'] = pd.to_numeric(df['her2'].fillna(df['her2'].median()), errors='coerce')
        df['tumor_subtype'] = df['tumor_subtype'].fillna('Unknown')

        unique_df = df.drop_duplicates(subset='patient_id', keep='first')

        unique_train_df, unique_validation_df = train_test_split(
                unique_df,
                train_size=0.80,
                test_size=0.20,
                stratify=unique_df['pcr'],
                random_state=0,
                shuffle=True,)
        # Filter the original DataFrame based on train_df patient_id and keep the first two image_id entries
        train_df = df[df['patient_id'].isin(unique_train_df['patient_id'])]
        # train_df = train_df.sort_values(by=['patient_id', 'image_id'])
        train_df = train_df.sort_values(by=['patient_id'])
        # train_df = train_df.groupby('patient_id').head(images_per_patient).reset_index(drop=True)

        validation_df = df[df['patient_id'].isin(unique_validation_df['patient_id'])]
        # validation_df = validation_df.sort_values(by=['patient_id', 'image_id'])
        validation_df = validation_df.sort_values(by=['patient_id'])
        # validation_df = validation_df.groupby('patient_id').head(images_per_patient).reset_index(drop=True) # taking the first two aquisitions of each patient
        train_df = train_df.drop_duplicates(subset='patient_id', keep='first')
        validation_df = validation_df.drop_duplicates(subset='patient_id', keep='first')
        #  Define the essential fields
        essential_fields = ["patient_id","image_id", "image_path", "pcr", "age", "tumor_subtype", "her2", "seg_path"]

        # Filter data to include only essential fields
        filtered_train_ds = [{key: item[key] for key in essential_fields} for item in train_df.to_dict("records")]
        filtered_validation_ds = [{key: item[key] for key in essential_fields} for item in validation_df.to_dict("records")]
        return filtered_train_ds, filtered_validation_ds
    def get_train_df(self):
        train_df = pd.read_csv(self.df_dir)

        essential_fields = ["patient_id","image_id", "image_path", "pcr", "age", "tumor_subtype", "her2", "seg_path"]

        # Filter data to include only essential fields
        filtered_train_ds = [{key: item[key] for key in essential_fields} for item in train_df.to_dict("records")]
        return filtered_train_ds
    def __len__(self):
        return len(self.dataset)
    


    def __getitem__(self, idx: int):

        data = self.dataset[idx]
        assert isinstance(data, dict), "data should be a dictionary"


        image_file = Path(data['image_path'])
        image_dir = image_file.parent
        images_path = sorted(os.listdir(image_dir))
        images_path = [os.path.join(image_dir, image) for image in images_path]
        img = [tio.ScalarImage(image) for image in images_path]
        label = tio.LabelMap(data['seg_path'])
        # tio_sub = tio.Subject(image=img, label=label)
        tio_sub = tio.Subject(
                **{f'image_{i}': im for i, im in enumerate(img)},
                label=label
            )
        tio_sub = self.preprocessing(tio_sub)
        # tio_sub = self.transforms(tio_sub)
        # return {'data': img.data.permute(0, -1, 1, 2)}
        return {'tio_sub': tio_sub,
                'data': data,

                }