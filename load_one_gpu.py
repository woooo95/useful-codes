#여러 gpu로 학습한 모델을 한 gpu로 로드해서 쓰는 방법

import torch

from infer import MultiMaX_BN_Unet
from collections import OrderedDict

checkpoint = "directory.pth"

device = torch.device("cuda:0")
model = "My Model"().to(device)

new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
    model.load_state_dict(new_state_dict)