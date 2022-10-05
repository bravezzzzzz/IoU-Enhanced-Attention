#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes

from .util.attention import MultiHeadSelfAttention
from torchvision.ops import box_iou


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION
        num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION

        # Init parameters.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.SparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features):

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes

        init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler)

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1_cls = nn.Linear(d_model, dim_feedforward)
        self.dropout_cls = nn.Dropout(dropout)
        self.linear2_cls = nn.Linear(dim_feedforward, d_model)
        self.linear1_reg = nn.Linear(d_model, dim_feedforward)
        self.dropout_reg = nn.Dropout(dropout)
        self.linear2_reg = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_cls = nn.LayerNorm(d_model)
        self.norm2_reg = nn.LayerNorm(d_model)
        self.norm3_cls = nn.LayerNorm(d_model)
        self.norm3_reg = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2_cls = nn.Dropout(dropout)
        self.dropout2_reg = nn.Dropout(dropout)
        self.dropout3_cls = nn.Dropout(dropout)
        self.dropout3_reg = nn.Dropout(dropout)

        self.proj_cls = nn.Linear(self.d_model, self.d_model)
        self.proj_reg = nn.Linear(self.d_model, self.d_model)
        self.norm_cls = nn.LayerNorm(self.d_model)
        self.norm_reg = nn.LayerNorm(self.d_model)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights


    def forward(self, features, bboxes, pro_features, pooler):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (nr_boxes, N, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        ious = []
        for i in range(N):
            iou = box_iou(bboxes[i], bboxes[i])
            ious.append(iou)
        ious = torch.stack(ious, dim=0)

        # iou enhanced self attention
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, pro_features, ious)
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features_cls, pro_features_reg = self.inst_interact(pro_features, roi_features)
        pro_features_cls = self.norm_cls(self.proj_cls(pro_features)) + self.dropout2_cls(pro_features_cls)
        pro_features_reg = self.norm_reg(self.proj_reg(pro_features)) + self.dropout2_reg(pro_features_reg)
        obj_features_cls = self.norm2_cls(pro_features_cls)
        obj_features_reg = self.norm2_reg(pro_features_reg)

        # obj_feature.
        obj_features_cls2 = self.linear2_cls(self.dropout_cls(self.activation(self.linear1_cls(obj_features_cls))))
        obj_features_reg2 = self.linear2_reg(self.dropout_reg(self.activation(self.linear1_reg(obj_features_reg))))
        obj_features_cls = (obj_features_cls) + self.dropout3_cls(obj_features_cls2)
        obj_features_reg = (obj_features_reg) + self.dropout3_reg(obj_features_reg2)
        obj_features_cls = self.norm3_cls(obj_features_cls)
        obj_features_reg = self.norm3_reg(obj_features_reg)

        obj_features_cls = obj_features_cls.transpose(0, 1).reshape(N * nr_boxes, -1)
        obj_features_reg = obj_features_reg.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = obj_features_cls.clone()
        reg_feature = obj_features_reg.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features_cls + obj_features_reg


    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        self.proj_cls1 = nn.Linear(self.hidden_dim, self.hidden_dim // 8)
        self.proj_cls2 = nn.Linear(self.hidden_dim // 8, self.hidden_dim)
        self.norm3_cls = nn.LayerNorm(self.hidden_dim)
        self.proj_reg1 = nn.Linear(self.hidden_dim, self.hidden_dim // 8)
        self.proj_reg2 = nn.Linear(self.hidden_dim // 8, self.hidden_dim)
        self.norm3_reg = nn.LayerNorm(self.hidden_dim)
        self.sigmoid = nn.Sigmoid()

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer_cls = nn.Linear(num_output, self.hidden_dim)
        self.norm4_cls = nn.LayerNorm(self.hidden_dim)
        self.out_layer_reg = nn.Linear(num_output, self.hidden_dim)
        self.norm4_reg = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        # dynamic channel weighting, DCW
        mask_cls = self.sigmoid(self.proj_cls2(self.proj_cls1(pro_features)))
        features_cls = torch.mul(features, mask_cls.transpose(1, 0))
        features_cls = self.norm3_cls(features_cls)
        features_cls = self.activation(features_cls)
        mask_reg = self.sigmoid(self.proj_reg2(self.proj_reg1(pro_features)))
        features_reg = torch.mul(features, mask_reg.transpose(1, 0))
        features_reg = self.norm3_reg(features_reg)
        features_reg = self.activation(features_reg)

        features_cls = features_cls.flatten(1)
        features_cls = self.out_layer_cls(features_cls)
        features_cls = self.norm4_cls(features_cls)
        features_cls = self.activation(features_cls)

        features_reg = features_reg.flatten(1)
        features_reg = self.out_layer_reg(features_reg)
        features_reg = self.norm4_reg(features_reg)
        features_reg = self.activation(features_reg)

        return features_cls, features_reg


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
