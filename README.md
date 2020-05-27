# CenterNet-FSAF-NMS-single-img
    在CenterNet的基础上加入了FSAF（Feature-Selective-Anchor-Free-Module-for-Single-Shot-Object-Detection）的思想，并且在输出检测结果之前用了NMS来过滤。
    将batch中的每个img拆分开，在loss最小的特征图上进行梯度回传。
    在VOC2007上做了实验，整体的AP相比原始的CenterNet上升了0.5，AR相比原始的AR上升了2.0。
