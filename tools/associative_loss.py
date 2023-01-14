import torch
import torch.nn as nn


class associative_loss(nn.Module):
    def __init__(self):
        super(associative_loss, self).__init__()

    def forward(self, feat_x, index_pos, index_neg):
        # loss_func = nn.MSELoss()
        loss_func = nn.CosineSimilarity(dim=-1)
        loss_total = []
        for t, idx_pos, idx_neg in zip(feat_x, index_pos, index_neg):

            # positive instances
            # first item
            cycle_feat = []
            for j in range(0, len(idx_pos), 2):
                if idx_pos[j] == -1:
                    break

                num = idx_pos[j + 1] - idx_pos[j]
                if num == 0:
                    feat_extract = t[idx_pos[j], :].view(-1, t.shape[1])
                elif num > 0:
                    feat_extract = t[idx_pos[j]:idx_pos[j + 1], :]
                else:
                    assert num >= 0, "num < 0"

                if feat_extract.shape[0] > 1:
                    cycle_feat.append(feat_extract.mean(dim=0).view(-1, feat_extract.shape[1]))
                elif feat_extract.shape[0] == 1:
                    cycle_feat.append(feat_extract)
                else:
                    print("feat_extract.shape[0]={}".format(feat_extract.shape[0]))
                    assert feat_extract.shape[0] >= 1, "feat_extract.shape[0] < 1"

            if len(cycle_feat)==0:
                print(1)

            cycle_feat = torch.stack(cycle_feat, dim=0).squeeze(1)

            cycle_feat_center = cycle_feat.mean(dim=0).view(-1, cycle_feat.shape[1])
            cycle_feat_center_first = cycle_feat_center.repeat(cycle_feat.shape[0], 1)

            loss_feat_1 = loss_func(cycle_feat, cycle_feat_center_first)
            loss_feat_1 = (1 - loss_feat_1).mean()

            # negative instances
            loss_feat_2 = []
            for k in range(0, len(idx_neg), 2):
                if idx_neg[k] == -1:
                    break

                num_neg = idx_neg[k + 1] - idx_neg[k]
                if num_neg == 0:
                    feat_extract_neg = t[idx_neg[k], :].view(-1, t.shape[1])
                elif num_neg > 0:
                    feat_extract_neg = t[idx_neg[k]:idx_neg[k + 1], :]
                else:
                    assert num_neg >= 0, "num < 0"

                if feat_extract_neg.shape[0] > 1:
                    non_cycle_feat_mean = feat_extract_neg.mean(dim=0).view(-1, feat_extract_neg.shape[1])
                elif feat_extract_neg.shape[0] == 1:
                    non_cycle_feat_mean = feat_extract_neg
                else:
                    print("feat_extract_neg.shape[0]={}".format(feat_extract_neg.shape[0]))
                    assert feat_extract_neg.shape[0] >= 1, "feat_extract_neg.shape[0] < 1"

                # loss_2 = torch.exp(- loss_func(non_cycle_feat_mean, cycle_feat_center))
                loss_2 = torch.exp(- (1 - loss_func(non_cycle_feat_mean, cycle_feat_center)))
                loss_feat_2.append(loss_2)

            loss_feat_2 = torch.stack(loss_feat_2).mean()

            loss_each_video = loss_feat_1 + loss_feat_2

            loss_total.append(loss_each_video)

        loss_total = torch.stack(loss_total).mean()

        return loss_total

