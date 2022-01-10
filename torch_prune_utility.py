import numpy as np
import torch



class PruneConfiguration():
    P1 = 83
    P2 = 92
    P3 = 99.1
    # P4 = 93
    P4 = 93
    # P1=P2=P3 =P4= 100-1e-9

    @staticmethod
    def display():
        print("P1 is %f" % PruneConfiguration.P1)
        print("P2 is %f" % PruneConfiguration.P2)
        print("P3 is %f" % PruneConfiguration.P3)
        print("P4 is %f" % PruneConfiguration.P4)


configuration = PruneConfiguration()
target_w = ['conv1/W_conv1', 'conv2/W_conv2', 'fc1/W_fc1', 'fc2/W_fc2']
# prune_percent = {'conv1/W_conv1': configuration.P1, 'conv2/W_conv2': configuration.P2, 'fc1/W_fc1': configuration.P3,
#                  'fc2/W_fc2': configuration.P4}
prune_percent = [configuration.P1,configuration.P2,configuration.P3,configuration.P4]


def get_configuration():
    return configuration


### sparsity
def projection(weight_arr, percent=10):
    '''sparsity contrains'''
    pcen = np.percentile(abs(weight_arr), percent)

    under_threshold = abs(weight_arr) < pcen
    weight_arr[under_threshold] = 0
    return weight_arr


def prune_weight(weight_arr, percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_arr = weight_arr.detach()
    pcen = np.percentile(abs(weight_arr), percent)
    # print("percentile " + str(pcen))
    under_threshold = abs(weight_arr) < pcen
    weight_arr[under_threshold] = 0
    above_threshold = abs(weight_arr) >= pcen
    return [weight_arr,above_threshold]

def apply_prune(model):
    thresh_mask =[]
    for idx, layer in enumerate(model.modules()):
        if idx > 0:
            # idx = 0 means CNN total structure, so we skip it
            print(idx, '\n\n->', layer)
            with torch.no_grad():
                before, before_num = (layer.weight!=0).sum(), layer.weight.abs().sum()
                # print("before pruning #non zero parameters " + str(before))
                weight_arr, mask = prune_weight(layer.weight.detach(),prune_percent[idx - 1])
                layer.weight.data = weight_arr
                after, after_num = (layer.weight != 0).sum(), layer.weight.abs().sum()
                # print("after pruning #non zero parameters " + str(after))
                print("pruned number %.8f pruned_weight_sum %.8f" %(before-after, before_num - after_num))
                thresh_mask.append(mask)
    return model, thresh_mask


def keep_mask(model, mask): # todo:how to clip gradience
    """record the prune results and apply to new model"""
    for idx, layer in enumerate(model.modules()):
        if idx > 0:
            with torch.no_grad():
                # print(mask[idx].size(), layer.weight.data.size())
                layer.weight.data = layer.weight.data * mask[idx - 1].float()
    return model



def apply_quantization(model, showing = True):
    thresh_mask =[]
    for idx, layer in enumerate(model.modules()):
        if idx > 0:
            # idx = 0 means CNN total structure, so we skip it
            if showing:print(idx, '->', layer)
            with torch.no_grad():
                before  = (layer.weight != 0) * (abs(layer.weight)!=1)
                if showing:print("before pruning #non ternary parameters " + str(before.sum()))
                layer.weight.data = quantization(layer.weight.detach(), percent=10)
                mask = layer.weight.data.detach().abs()
                after = (layer.weight != 0) * (abs(layer.weight) != 1)
                if showing:print("after pruning #non ternary parameters " + str(after.sum()))
                # before, before_num = (layer.weight!=0).sum(), layer.weight.abs().sum()
                # print("before pruning #non zero parameters " + str(before))
                # weight_arr, mask = prune_weight(layer.weight.detach(),prune_percent[idx - 1])
                # layer.weight.data = weight_arr
                # after, after_num = (layer.weight != 0).sum(), layer.weight.abs().sum()
                # print("pruned number %.8f pruned_weight_sum %.8f" %(before-after, before_num - after_num))
                thresh_mask.append(mask)
    return model, thresh_mask

def quantization(weight_arr, percent,max_epoch = 5):
    # according to Eq 13
    with torch.no_grad():
        V = weight_arr.view(-1,1) # convert into a vector
        # approximate V
        Q = torch.ones_like(V)
        for i in range(max_epoch):
            alpha = V.t().mm(Q) / Q.t().mm(Q)
            Q = Ternary(V / alpha)
    # print(Q)
    return Q.view(weight_arr.size()) * alpha


def Ternary(weight_arr):
    weight_arr[abs(weight_arr) < 0.5] = 0
    weight_arr[weight_arr>0.5] = 1
    weight_arr[weight_arr < -0.5] = -1
    return weight_arr

# def apply_prune(dense_w, sess):
#     # returns dictionary of non_zero_values' indices
#     dict_nzidx = {}
#     for target_name in target_w:
#         print("at weight " + target_name)
#         weight_arr = sess.run(dense_w[target_name])
#         print("before pruning #non zero parameters " + str(np.sum(weight_arr != 0)))
#         before = np.sum(weight_arr != 0)
#         mask, weight_arr_pruned = prune_weight(weight_arr, target_name)
#         after = np.sum(weight_arr_pruned != 0)
#         print("pruned " + str(before - after))
#
#         print("after prunning #non zero parameters " + str(np.sum(weight_arr_pruned != 0)))
#         sess.run(dense_w[target_name].assign(weight_arr_pruned))
#         dict_nzidx[target_name] = mask
#     return dict_nzidx

# def apply_prune_on_grads(grads_and_vars,dict_nzidx):
#
#   for key, nzidx in dict_nzidx.items():
#     count = 0
#     for grad, var in grads_and_vars:
#       if var.name == key+":0":
#         nzidx_obj = tf.cast(tf.constant(nzidx), tf.float32)
#         grads_and_vars[count] = (tf.multiply(nzidx_obj, grad), var)
#       count += 1
#   return grads_and_vars


