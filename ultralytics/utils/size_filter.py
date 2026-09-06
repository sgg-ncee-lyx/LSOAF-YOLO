import torch

def apply_size_filter(pred_dict, area_range=None, img_shape=None):
    """
    针对 YOLOv11 预测字典的面积过滤器
    img_shape: (height, width) 原始图片尺寸
    """
    if area_range is None or pred_dict is None or len(pred_dict.get('bboxes', [])) == 0:
        return pred_dict

    # 1. 复制坐标，防止修改原始引用
    bboxes = pred_dict['bboxes'].clone()

    # 注意：YOLO v11 的 pred_dict['bboxes'] 通常是 xyxy 格式
    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]

    # 如果此时 ws.max() 很小（比如 < 1），说明是归一化坐标，必须乘上 img_shape
    if ws.max() < 2.0 and img_shape is not None:
        ws *= img_shape[1]  # width
        hs *= img_shape[0]  # height

    areas = ws * hs
    keep = (areas >= area_range[0]) & (areas < area_range[1])


    # 5. 过滤所有字段
    new_dict = {}
    for k, v in pred_dict.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == keep.shape[0]:
            new_dict[k] = v[keep]
        else:
            new_dict[k] = v

    return new_dict