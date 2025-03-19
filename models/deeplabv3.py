from mmengine.model import BaseModel
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
import torch

class MMDeeplabV3(BaseModel):

    def __init__(self, num_classes):
        super().__init__()
        self.deeplab = deeplabv3_resnet50()
        self.deeplab.classifier[4] = torch.nn.Conv2d(
            256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, imgs, data_samples=None, mode='tensor'):
        #fea=self.deeplab.backbone(imgs)
        x = self.deeplab(imgs)['out']
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, data_samples['labels'].long())}
        elif mode == 'predict':
            return x, data_samples

    def select_step(self, data):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs