import torch
from torch import nn
from efficientdet.loss import FocalLoss
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, to_bbox_detections


class ModelWithRPN(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.cfg = cfg.clone()
        self.debug = debug

        self.criterion = FocalLoss()
        self.pre_nms_thresh = 0.05
        self.nms_thresh = 0.5
        self.fpn_post_nms_top_n = 100

        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.image_size = self.input_sizes[self.cfg.EFFICIENTNET.COEF]

    def forward(self, imgs, annotations, regression, classification, anchors, obj_list=None):
        if self.training:
            return self._forward_train(anchors, classification, regression, annotations, imgs, obj_list=obj_list)
        else:
            return self._forward_test(anchors, classification, regression, imgs)

    def _forward_train(self, anchors, classification, regression, annotations, imgs, obj_list=None):
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)

        losses = {
            "loss_retina_cls": cls_loss,
            "loss_retina_reg": reg_loss,
        }

        detections = None
        if self.cfg.MODEL.MASK_ON:
            with torch.no_grad():
                # detections = self.box_selector_train(
                #     anchors, box_cls, box_regression
                # )
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                out = postprocess(imgs.detach(), anchors, regression, classification, regressBoxes, clipBoxes,
                                  self.pre_nms_thresh, self.nms_thresh)
                detections = to_bbox_detections(out, img_size = self.image_size, fpn_post_nms_top_n=self.fpn_post_nms_top_n)
        return (anchors, detections), losses

    def _forward_test(self, anchors, classification, regression, imgs):
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        #TODO: currently the limit for pre_nms_top_n is not set
        #TODO: retinamask does nms per class(label) but Yet-Anotehr does nms all togehter
        out = postprocess(imgs.detach(), anchors, regression, classification, regressBoxes, clipBoxes,
                          self.pre_nms_thresh, self.nms_thresh)
        boxes = to_bbox_detections(out, img_size = self.image_size, fpn_post_nms_top_n=self.fpn_post_nms_top_n)

        return (anchors, boxes), {}


def build_rpn(cfg):
    return ModelWithRPN(cfg)
