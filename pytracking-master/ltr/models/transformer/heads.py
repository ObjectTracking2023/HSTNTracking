import torch
import torch.nn as nn
import ltr.models.layers.filter as filter_layer
from ltr.models.stdomain.spatial import SEModule,NONLocalBlock2D
from ltr.models.stdomain.temporal import CTFM



def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(1, outplanes),
        nn.ReLU(inplace=True),
    ]
    return layers


class Head(nn.Module):
    """
    """
    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor,
                 separate_filters_for_cls_and_bbreg=False):
        super().__init__()

        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

        self.channelatt = SEModule(256 * 3, 3)
        self.spatialatt = NONLocalBlock2D(in_channels=256, inter_channels=256)
        self.memory = CTFM()

    def forward(self, train_feats, test_feats, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        cls_filter_list = []
        reg_filter_list = []
        test_feat_list = []

        a = train_feats[0].reshape(-1, num_sequences, *train_feats[0].shape[-3:])
        b = train_feats[1].reshape(-1, num_sequences, *train_feats[1].shape[-3:])
        c = train_feats[2].reshape(-1, num_sequences, *train_feats[2].shape[-3:])

        assert ((a.shape[0] == 2) and (b.shape[0] == 2) and (c.shape[0] == 2))

        template = []
        template.append(a[0:1,:,:,:,:].reshape(num_sequences, *train_feats[0].shape[-3:]))
        template.append(a[0:1,:,:,:,:].reshape(num_sequences, *train_feats[0].shape[-3:]))
        template.append(a[0:1,:,:,:,:].reshape(num_sequences, *train_feats[0].shape[-3:]))

        memory = []
        memory.append(a[1:2,:,:,:,:].reshape(num_sequences, *train_feats[0].shape[-3:]))
        memory.append(a[1:2,:,:,:,:].reshape(num_sequences, *train_feats[0].shape[-3:]))
        memory.append(a[1:2,:,:,:,:].reshape(num_sequences, *train_feats[0].shape[-3:]))


        train_feats = self.memory(template, memory, test_feats)

        train_feats[0] = torch.cat((train_feats[0], template[0]), dim=0)
        train_feats[1] = torch.cat((train_feats[1], template[1]), dim=0)
        train_feats[2] = torch.cat((train_feats[2], template[2]), dim=0)

        for idx, (train_feat, test_feat) in enumerate(zip(train_feats, test_feats), start=2):
            # if train_feat.dim() == 5:
            #     train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
            # if test_feat.dim() == 5:
            #     test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

            # Extract features

            train_feat = self.extract_head_feat(train_feat, num_sequences)
            test_feat = self.extract_head_feat(test_feat, num_sequences)



        # Train filter
            cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)
            test_feat = test_feat_enc.reshape(-1, *test_feat_enc.shape[-3:])
            cls_filter_list.append(cls_filter)
            reg_filter_list.append(breg_filter)
            test_feat_list.append(test_feat)

        cls_feature = self.channelatt(torch.cat(cls_filter_list, 1))
        reg_feature = self.channelatt(torch.cat(reg_filter_list, 1))
        test_feature = self.channelatt(torch.cat(test_feat_list, 1))

        cls_feature = [cls_feature[:, 0:256, :, :], cls_feature[:, 256:512, :, :], cls_feature[:, 512:768, :, :]]
        reg_feature = [reg_feature[:, 0:256, :, :], reg_feature[:, 256:512, :, :], reg_feature[:, 512:768, :, :]]
        test_feature = [test_feature[:, 0:256, :, :], test_feature[:, 256:512, :, :], test_feature[:, 512:768, :, :]]

        cls_feature = self.spatialatt(cls_feature[0], cls_feature[1], cls_feature[2])
        reg_feature = self.spatialatt(reg_feature[0], reg_feature[1], reg_feature[2])
        test_feature = self.spatialatt(test_feature[0], test_feature[1], test_feature[2])
        test_feature = test_feature.reshape(1, num_sequences, 256, *test_feat_enc.shape[-2:])


        # fuse encoder and decoder features to one feature map
        target_scores = self.classifier(test_feature, cls_feature)

        # compute the final prediction using the output module
        bbox_preds = self.bb_regressor(test_feature, reg_feature)
        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        # feat:  Input feature maps. Dims (images_in_sequence, sequences, feat_dim, H, W).
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights

        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc \
            = self.filter_predictor.predict_cls_bbreg_filters_parallel(
            train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs
        )

        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class LinearFilterClassifier(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)


class DenseBoxRegressor(nn.Module):
    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter

        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)

        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape

        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter

        attention = filter_layer.apply_filter(feat, filter_proj) # (nf, ns, h, w)
        feats_att = attention.unsqueeze(2)*feat # (nf, ns, c, h, w)

        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1])) # (nf*ns, c, h, w)

        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0) # (nf*ns, 4, h, w)
        return ltrb
