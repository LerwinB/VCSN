from mmseg.registry import DATASETS
from .basesegdataset import BaseCDDataset


@DATASETS.register_module()
class HilsatDataset(BaseCDDataset):

    METAINFO = dict(
        classes=( 'background', 'panel', 'aerial','spout','camera','panel support','star sensor','docking','body'),
        palette=[[0, 0, 0],[255, 0, 0], [0, 255, 0],[255, 255, 0],[0, 0, 255],[255, 0, 255],[0, 0, 124],[202, 202, 202],[255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,reduce_zero_label=reduce_zero_label, **kwargs)

    