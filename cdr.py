import numpy as np

import cv2
import torch
import torch.nn.functional as F

from skimage import feature
from skimage.segmentation import slic

def cdr(inputs, gt, kernel_size = 5, target_region_mask = None):
    """
    inputs: Target Image(Lab) with H*W*C
    gt: GT Image(Lab) with H*W*C
    kernel_size: Area of mesure the CDR
    target_region_mask: Region of interest (Binary Scribble in the paper)  
    return: CDR ratio for each a & b channel
    """
    if type(org_inputs) == np.ndarray:
        inputs = torch.Tensor(inputs)
        gt = torch.Tensor(gt)
    pad_size = (kernel_size // 2 + 1, kernel_size // 2 + 1, kernel_size // 2 + 1, kernel_size // 2 + 1)
    gt_a = gt[: ,: , 1].unsqueeze(-1).repeat(1 ,1, 3)
    gt_b = gt[: ,: , 2].unsqueeze(-1).repeat(1, 1, 3)
    inputs_a = inputs[: ,: , 1].unsqueeze(-1).repeat(1, 1, 3)
    inputs_b = inputs[: ,: , 2].unsqueeze(-1).repeat(1, 1, 3)

    gt_a_slic = torch.Tensor(slic(gt_a.double().numpy(), n_segments=250, compactness=10, sigma=1,
                                    start_label=1))
    gt_b_slic = torch.Tensor(slic(gt_b.double().numpy(), n_segments=250, compactness=10, sigma=1,
                                    start_label=1))
    inputs_a_slic = torch.Tensor(slic(inputs_a.double().numpy(), n_segments=250, compactness=10, sigma=1,
                                    start_label=1))
    inputs_b_slic = torch.Tensor(slic(inputs_b.double().numpy(), n_segments=250, compactness=10, sigma=1,
                                    start_label=1))

    # Add the padding
    gt_a_slic = F.pad(gt_a_slic, pad_size, "constant", 0)
    gt_b_slic = F.pad(gt_b_slic, pad_size, "constant", 0)
    inputs_a_slic = F.pad(inputs_a_slic, pad_size, "constant", 0)
    inputs_b_slic = F.pad(inputs_b_slic, pad_size, "constant", 0)

    canny_a = torch.Tensor(feature.canny(gt_a[:,:,0].numpy(), sigma=1.2, high_threshold=0.7, low_threshold=0.2, use_quantiles=0.4))
    canny_b = torch.Tensor(feature.canny(gt_b[:,:,0].numpy(), sigma=1.2, high_threshold=0.7, low_threshold=0.2, use_quantiles=0.4))

    if target_region_mask is None:
        canny_a = F.pad(canny_a, pad_size, "constant", 0)
        canny_b = F.pad(canny_b, pad_size, "constant", 0)
    else:
        canny_a = F.pad(canny_a * target_region_mask, pad_size, "constant", 0)
        canny_b = F.pad(canny_a * target_region_mask, pad_size, "constant", 0)

    canny_a_coor = canny_a.nonzero()
    canny_b_coor = canny_b.nonzero()

    cdr_a, cdr_b = 0, 0  
    for num_edge_a, coor in enumerate(range(canny_a_coor.shape[0])):
        h, w = canny_a_coor[coor][-2], canny_a_coor[coor][-1]

        gt_sc_a = gt_a_slic[h - kernel_size + 1:h + kernel_size,
                    w - kernel_size + 1:w + kernel_size] != gt_a_slic[h, w]

        inputs_sc_a = inputs_a_slic[h - kernel_size + 1:h + kernel_size,
                    w - kernel_size + 1:w + kernel_size] == inputs_a_slic[h, w]

        if gt_sc_a.sum() != 0:
            cdr_a += 1 - float((gt_sc_a * inputs_sc_a).sum()) / float(gt_sc_a.sum())  
        else:
            cdr_a += 1
    cdr_a /=(num_edge_a+1)

    for num_edge_b, coor in enumerate(range(canny_b_coor.shape[0])):
        h, w = canny_b_coor[coor][-2], canny_b_coor[coor][-1]

        gt_sc_b = gt_b_slic[h - kernel_size + 1:h + kernel_size,
                    w - kernel_size + 1:w + kernel_size] != gt_b_slic[h, w]

        inputs_sc_b = inputs_b_slic[h - kernel_size + 1:h + kernel_size,
                    w - kernel_size + 1:w + kernel_size] == inputs_b_slic[h, w]

        if gt_sc_b.sum() != 0:
            cdr_b += 1 - float((gt_sc_b * inputs_sc_b).sum()) / float(gt_sc_b.sum())  
        else:
            cdr_b += 1
    cdr_b /=(num_edge_b+1)

    return cdr_a, cdr_b 
    
if __name__ == "__main__":
    example_gt = cv2.cvtColor(cv2.imread('./example/gt.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
    org_inputs = cv2.cvtColor(cv2.imread('./example/org_inputs.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
    enh_inputs = cv2.cvtColor(cv2.imread('./example/enh_inputs.png', cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)

    cdr_a, cdr_b = cdr(inputs = org_inputs, gt = example_gt)
    print(f'Org_CDR_a:{cdr_a:.3f} Org_CDR_b:{cdr_b:.3f}')
    cdr_a, cdr_b = cdr(inputs = enh_inputs, gt = example_gt)
    print(f'Enh_CDR_a:{cdr_a:.3f} Enh_CDR_b:{cdr_b:.3f}')
