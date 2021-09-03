## Yolo Data Balancer Augmentation

- Libs Requirements
```
pip install albumentations
pip install opencv-python
pip install imgaug
pip install tqdm
```
## Data Structure
```
.data
├── obj_train_data
│   └── [images and txt]
├── obj.data
├── obj.names
└── train.txt
```

## Use methods
```
from data_balance import Data_balancer

# instance class
data_balancer = Data_Balancer()

# detect and count all classes
data_balancer.detect_class("path")

# choose class to augmentation
data_balancer.balance_class()
```
## Results Examples
```
detect_class method returns example:
===================================================================================
{'Costela': 0, 'Osso': 0, 'osso_bom': 961, 'osso_ruim': 290, 'total_objects': 1251}
===================================================================================

balance_class method returns example:
======================================================================
New Classes
===================================================================================
{'Costela': 0, 'Osso': 0, 'osso_bom': 1012, 'osso_ruim': 579, 'total_objects': 1591}
===================================================================================
Writing new_train.txt file...
OK!
```
---

## Future Steps
- include raise and assert
- include error handling
- automatic file finder
- include argparser
---