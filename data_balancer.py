from typing import Any
from glob import glob
import os
import cv2
from tqdm import tqdm
import albumentations as A

class Data_Balancer():
    """
    This Class works only on Yolo Dataset Format
    """
    def __init__(self) -> None:
        self.data_dict = dict()
        self.class_dict = dict()
        self.train_txt = []


    def _read_data_file(self, path: str)->list:
        if "obj.data" in os.listdir(path):
            r_file = "obj.data"
            file = open(os.path.join(path, r_file), "r").readlines()
            file = [row.split("\n")[0] for row in file]
            file.pop(-1)
            num_class, file_list, names = file
            num_class = num_class.split(" ")[-1]
            file_list = file_list.split(" ")[-1]
            names = names.split(" ")[-1]
            return num_class, file_list, names
        else:
            raise f"obj.data doens't exist in {path}"
    
    def _read_txt_file(self, path: str)->Any:
        file = open(path, "r").readlines()
        file = [row.split("\n")[0] for row in file]
        return file

    def _split_to_id(self, path: str)->list:
        file_list = glob(os.path.join(path, "obj_train_data/*.txt"))
        id_list = []
        if file_list:
            file_list_opened = [open(file, "r").readlines() for file in file_list]
            for idx, label in enumerate(file_list_opened):
                if label:
                    if len(label)>1:
                        label = [int(row.split(" ")[0]) for row in label]
                        id_list.extend(label)
                    else:
                        id_list.append(int(label[0].split(" ")[0]))
            self.path = path
            return id_list
        else:
            raise f"Can't read txt file in {path}"
    
    def _load_and_aug(self, class_id: int):
        file_list = glob(os.path.join(self.path, "obj_train_data/*.txt"))
        base_dir = os.path.join(self.path, "obj_train_data/")
        if file_list:
            file_list_opened = [open(file, "r").readlines() for file in file_list]
            for idx, label in tqdm(enumerate(file_list_opened)):
                try:
                    file_name = os.path.basename(file_list[idx]).split(".txt")[0]
                    file_name = file_name
                    img_path = base_dir + file_name + ".jpg"
                    self.train_txt.append(img_path)
                    if label:
                        if len(label)>1:
                            bbox = [
                                [float(row.split(" ")[1]), float(row.split(" ")[2]), float(row.split(" ")[3]), float(row.split(" ")[4])]
                                for row in label
                                ]
                            label = [int(row.split(" ")[0]) for row in label]
                            temp_class_list = []
                            for class_number in label:
                                temp_class_list.append({class_number: self.class_dict[class_number]})
                            if class_id in label:
                                file_name = os.path.basename(file_list[idx]).split(".txt")[0]
                                file_name = file_name
                                img_path = base_dir + file_name + ".jpg"
                                transform = self._apply_aug(img_path, bbox, temp_class_list)
                                self._write_aug(transform, base_dir+file_name)
                        else:
                            if class_id == int(label[0].split(" ")[0]):
                                bbox = [
                                [float(row.split(" ")[1]), float(row.split(" ")[2]), float(row.split(" ")[3]), float(row.split(" ")[4])]
                                for row in label
                                ]
                                label = [int(row.split(" ")[0]) for row in label]
                                temp_class_list = []
                                for class_number in label:
                                    temp_class_list.append({class_number: self.class_dict[class_number]})
                                file_name = os.path.basename(file_list[idx]).split(".txt")[0]
                                file_name = file_name
                                img_path = base_dir + file_name + ".jpg"
                                transform = self._apply_aug(img_path, bbox, temp_class_list)
                                self._write_aug(transform, base_dir+file_name)
                
                except Exception as e:
                    print(e)
                    print(file_list[idx])
                    continue

    def _load_transform(self):
        transform = A.Compose([A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])], bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]))
        return transform


    def _apply_aug(self, img_path: str, bbox: list, temp_class_dict: dict):
        transform = self._load_transform()
        img = cv2.imread(img_path)
        try:
            transform = transform(image=img, bboxes=bbox, category_ids=temp_class_dict)
        except Exception as e:
            print(e)
            print(img_path)
        return transform

    def _write_aug(self, transform, output_dir):
        cv2.imwrite(output_dir+"_aug.jpg", transform["image"])
        self.train_txt.append(output_dir+"_aug.jpg")
        bbox_list = transform["bboxes"]
        class_id = transform['category_ids']
        class_id = [[*key] for key in class_id]
        class_id = [str(key[0]) for key in class_id]
        with open(output_dir+"_aug.txt", "w") as f:
            for idx, box in enumerate(bbox_list):
                box = list(box)
                box = [str(row) for row in box]
                box = [class_id[idx]]+box
                box = " ".join(box)
                box = box+"\n"
                f.writelines(box)
        f.close()
            
    def _write_train(self):
        print("Writing new_train.txt file...")
        with open(self.path+"/new_train.txt", "w") as f:
            f.writelines("\n".join(self.train_txt))
        f.close()
        print("OK!")

    def _id_to_class(self, id_list:list, class_dict:dict)->Any:
        data_dict = dict()
        total = 0
        for keys, ids in class_dict.items():
            data_dict.update({ids: id_list.count(keys)})
            total+=id_list.count(keys)
        data_dict.update({"total_objects": total})
        return data_dict

    def _choose_class(self)->str:
        if self.data_dict is not None:
            print()
            classes = [t for t in self.data_dict.keys() if t != "total_objects"]
            self.classes = classes.copy()
            self.classes = [classe.lower() for classe in self.classes]
            classes = "\n".join(classes)
            idx = str(input(f"Which class do you want to balance?\n{classes}\n"))
            return idx.lower()
        else:
            return 

    def _create_class_dict(self, num_class: int, names: list)->dict:
        class_dict = dict()
        for i in range(0, num_class):
            class_dict.update({i: names[i]})
        self.class_dict = class_dict.copy()
        return class_dict

    def detect_class(self, path: str)-> Any:
        num_class, file_list, names = self._read_data_file(path)
        num_class = int(num_class)
        names = self._read_txt_file(names)
        id_list = self._split_to_id(path)
        class_dict = self._create_class_dict(num_class, names)
        data_dict = self._id_to_class(id_list, class_dict)
        self.data_dict = data_dict.copy()
        print("="*70)
        print(self.data_dict)
        print("="*70)

    def balance_class(self)-> Any:
        if self.data_dict is not None:
            key = self._choose_class()
            if key in self.classes:
                if self.data_dict[key]>10:
                    class_id = [idx for idx, value in self.class_dict.items() if value==key]
                    class_id = class_id[0]
                    self._load_and_aug(class_id)
                    print()
                    print("="*70)
                    print("New Classes")
                    self.detect_class(self.path)
                    self._write_train()
                else:
                    print("Please Increase the number of samples to more than 10!")
            else:
                print("Please choose an existing class!")
        else:
            print("Use detect_class method to find unbalaced class!")

teste = Data_Balancer()
teste.detect_class("data")
teste.balance_class()
