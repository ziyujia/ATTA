import logging
from torch import nn
import torch
import torch.optim as optim
from model_modified import SleepTTA
from functools import reduce
from torch.utils.data import DataLoader
from preprocess_sleepedf import get_double_eeg_datasets,get_double_eog_datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, confusion_matrix
import torch.nn.functional as F
import os
from memory import Memory
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    """
        parser arguments and setting log formats
        return: the arguments after parse
    """
    parser = argparse.ArgumentParser(description="Set hyperparameters for the model.")
    parser.add_argument("--data_path", '-d', type=str, default="../data/sleepedf/sleep-cassette/prepared", help="Path to the data.")
    parser.add_argument("--pth_dir", '-p', type=str, default="SalientSleepNet/pretrained_model",
                        help="Directory to the pre-trained model.")
    parser.add_argument("--batch_size", '-b', type=int, default=20, help="Batch size for training.")
    parser.add_argument("--window_size", '-w', type=int, default=35, help="Window size.")
    parser.add_argument("--lr", '-l',type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--optimizer_method", '-o', type=str, default="Adam", choices=["Adam", "SGD", "RMSprop"], help="Optimizer method.")
    parser.add_argument("--momentum", '-m', type=float, default=2e-2, help="Momentum for the update of retained model.")
    parser.add_argument("--memory_size", type=int, default=560, help="Size of the memory bank.")
    parser.add_argument("--retrieval_size", '-r', type=int, default=70, help=" the number of feature vectors retrieved from the memory bank.")
    parser.add_argument("--log_file", type=str, default='./log/SHHSto153/SHHSto153.txt', help="Path to the log file.")
    parser.add_argument("--beta", type=float, default=1e3, help="Beta for similarity metric.")
    parser.add_argument("--title_name", '-t', type=str, default="matrix",help="Matrix picture title.")
    args = parser.parse_args()
    return args

args = get_args()
logger = logging.getLogger(__name__)
data_path, pth_dir, batch_size, window_size, lr, optimizer_method, \
    momentum, memory_size, retrieval_size, log_file, beta, title_name= args

double_eeg_dataset_list=get_double_eeg_datasets(data_path,stride=35)
double_eog_dataset_list=get_double_eog_datasets(data_path,stride=35)

for eeg_dataset in double_eeg_dataset_list:
    eeg_dataset.normalization()
for eog_dataset in double_eog_dataset_list:
    eog_dataset.normalization()

def evaluate():
    lr0 = args.lr
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    steps = 1
    assert steps > 0, "Warning : steps out of range!!!"
    eeg_model_path = pth_dir + '/model_shhs_eeg.pth' # Path where the model is saved.
    eog_model_path = pth_dir + '/model_shhs_eog.pth' # Pretrained on shhs,tested on sleepedf

    eeg_model_a, eeg_optimizer_a, eeg_model_r = SingleConfig(path=eeg_model_path, device = device)
    eog_model_a, eog_optimizer_a, eog_model_r = SingleConfig(path=eog_model_path, device = device)

    eeg_dataset_list_total = data_split(double_eeg_dataset_list, total_num=153)
    eog_dataset_list_total = data_split(double_eog_dataset_list, total_num=153)

    test_num = 0
    """        
        For eeg and eog channels respectively:
        1. For all the samples in the batch (b samples in total), first input them into the adaptive-retained model for each of the two channels to get the feature layer output (taking an arithmetic average of the adaptive model and the retained model) q_m=h(x) 
        2. Call get_neighbours function to get the weighted average model output pt_m (bi=0...b) obtained after finding k (k is retrieval_size) neighbors from all samples.(m denotes eeg or eog)
        3. Calculate the KL divergence based on the output f(x,θ) of the current batch of the model and compute to get the learning rate for updating the learning rate in the subsequent process
        4. Redo the prediction
        5. Update memory bank
    """
    metrics_all_subject = {}
    metrics_all_subject['acc'], metrics_all_subject['f1'], metrics_all_subject['prec'], metrics_all_subject['rec'] = [], [], [], []
    label_all_subject = {}

    label_all_subject['t_label'], label_all_subject['p_label'] = [], []
    memory_eeg,memory_eog=Memory(memory_size),Memory(memory_size)
    for eeg_test_set, eog_test_set in zip(eeg_dataset_list_total, eog_dataset_list_total):

        eeg_test_set = reduce(lambda x, y: x + y, eeg_test_set)
        eog_test_set = reduce(lambda x, y: x + y, eog_test_set)
        eeg_test_loader = DataLoader(eeg_test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        eog_test_loader = DataLoader(eog_test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        # Record batch normalization layer information for eeg and eog
        eeg_prev_bn_params_r = eeg_model_r.state_dict()
        eog_prev_bn_params_r = eog_model_r.state_dict()
        eeg_bn_params_r, eeg_bn_names_r = collect_params(eeg_model_r)
        eog_bn_params_r, eog_bn_names_r = collect_params(eog_model_r)

        # Forward propagation
        running_loss = 0.0
        acc, f1, prec, rec = [], [], [], []
        acc1, acc2 = [], []
        f1_1, f1_2 = [], []
        
        for (x_eeg, y_eeg), (x_eog, y_eog) in zip(eeg_test_loader, eog_test_loader):

            for step in range(1, steps + 1):
                x_eeg = x_eeg.to(device)
                y_eeg = y_eeg.to(device)
                x_eog = x_eog.to(device)
                y_eog = y_eog.to(device)

                true_label_eeg = y_eeg
                true_label_eog = y_eog
                assert torch.all(true_label_eeg == true_label_eog), "Warning : labeling inconsistencies between eeg and eog!!!"

                # intra-modal output for eeg
                eeg_output_a = eeg_model_a(x_eeg)
                eeg_output_a = eeg_output_a.transpose(1, 2)
                eeg_output_a = eeg_output_a.reshape(-1, 5)
                eeg_output_a = F.softmax(eeg_output_a, dim=1)

                eeg_output_r = eeg_model_r(x_eeg)
                eeg_output_r = eeg_output_r.transpose(1, 2)
                eeg_output_r = eeg_output_r.reshape(-1, 5)
                eeg_output_r = F.softmax(eeg_output_r, dim=1)

                eeg_output = (eeg_output_r + eeg_output_a) / 2

                # intra-modal output for eog
                eog_output_a = eog_model_a(x_eog)
                eog_output_a = eog_output_a.transpose(1, 2)
                eog_output_a = eog_output_a.reshape(-1, 5)
                eog_output_a = F.softmax(eog_output_a, dim=1)

                eog_output_r = eog_model_r(x_eog)
                eog_output_r = eog_output_r.transpose(1, 2)
                eog_output_r = eog_output_r.reshape(-1, 5)
                eog_output_r = F.softmax(eog_output_r, dim=1)

                eog_output = (eog_output_r + eog_output_a) / 2

                true_label_eeg = true_label_eeg.reshape(true_label_eeg.shape[0] * true_label_eeg.shape[1])

                # similarity metric for eeg and eog
                sim_eeg = sim(eeg_output_a, eeg_output_r)
                sim_eog = sim(eog_output_a, eog_output_r)

                pseudo_label = balanced(sim_eeg, sim_eog, eeg_output, eog_output)

                with torch.no_grad():
                    keys_eeg = (eeg_model_r(x_eeg, feature_extract=True) + eeg_model_a(x_eeg, feature_extract=True)) / 2
                    keys_eog = (eog_model_r(x_eog, feature_extract=True) + eog_model_a(x_eog, feature_extract=True)) / 2
                keys_eeg=keys_eeg.detach().cpu().numpy()
                keys_eog=keys_eog.detach().cpu().numpy()
                p_eeg=memory_eeg.get_neighbours(keys_eeg, retrieval_size)
                p_eog=memory_eog.get_neighbours(keys_eog, retrieval_size)

                if len(p_eeg)!=0 or len(p_eog)!=0:
                    alpha_eeg=similar(p_eeg, eeg_output)
                    alpha_eog=similar(p_eog, eog_output)
                    alpha=0.5*(alpha_eeg+alpha_eog)
                    # update learning rate
                    lr=lr0*alpha
                    for param_group in eeg_optimizer_a.param_groups:
                        param_group['lr'] = lr
                    for param_group in eog_optimizer_a.param_groups:
                        param_group['lr'] = lr

                loss = softmax_entropy(eeg_output, eog_output, pseudo_label)

                # backward propagation
                loss.backward()
                running_loss += loss
                eeg_optimizer_a.step()
                eog_optimizer_a.step()
                eeg_optimizer_a.zero_grad()
                eog_optimizer_a.zero_grad()
                # update the retained model
                for i in range(len(eeg_bn_params_r)):
                    name = eeg_bn_names_r[i]
                    eeg_bn_params_a = eeg_model_a.state_dict()
                    eeg_prev_bn_params_r[name] = momentum * eeg_bn_params_a[name] + (1 - momentum) * \
                                                 eeg_prev_bn_params_r[name]
                    eeg_bn_params_r[i].data = eeg_prev_bn_params_r[name]

                for i in range(len(eog_bn_params_r)):
                    name = eog_bn_names_r[i]
                    eog_bn_params_a = eog_model_a.state_dict()
                    eog_prev_bn_params_r[name] = momentum * eog_bn_params_a[name] + (1 - momentum) * \
                                                 eog_prev_bn_params_r[name]
                    eog_bn_params_r[i].data = eog_prev_bn_params_r[name]
                # Redo the prediction.
                with torch.no_grad():
                    eeg_output_a = eeg_model_a(x_eeg)
                    eeg_output_a = eeg_output_a.transpose(1, 2)
                    eeg_output_a = eeg_output_a.reshape(-1, 5)
                    eeg_output_a = F.softmax(eeg_output_a, dim=1)

                    eeg_output_r = eeg_model_r(x_eeg)
                    eeg_output_r = eeg_output_r.transpose(1, 2)
                    eeg_output_r = eeg_output_r.reshape(-1, 5)
                    eeg_output_r = F.softmax(eeg_output_r, dim=1)

                    eeg_output = (eeg_output_r + eeg_output_a) / 2

                    eog_output_a = eog_model_a(x_eog)
                    eog_output_a = eog_output_a.transpose(1, 2)
                    eog_output_a = eog_output_a.reshape(-1, 5)
                    eog_output_a = F.softmax(eog_output_a, dim=1)

                    eog_output_r = eog_model_r(x_eog)
                    eog_output_r = eog_output_r.transpose(1, 2)
                    eog_output_r = eog_output_r.reshape(-1, 5)
                    eog_output_r = F.softmax(eog_output_r, dim=1)

                    eog_output = (eog_output_r + eog_output_a) / 2

                    # similarity metric for eeg and eog.
                    sim_eeg = sim(eeg_output_a, eeg_output_r)
                    sim_eog = sim(eog_output_a, eog_output_r)

                    # Obtaining pseudo-labels based on the balanced modal fusion strategy
                    pseudo_label = balanced(sim_eeg, sim_eog, eeg_output, eog_output)
                    

                    keys_eeg = (eeg_model_r(x_eeg, feature_extract=True) + eeg_model_a(x_eeg, feature_extract=True)) / 2
                    keys_eog = (eog_model_r(x_eog, feature_extract=True) + eog_model_a(x_eog, feature_extract=True)) / 2
                    keys_eeg=keys_eeg.detach().cpu().numpy()
                    keys_eog=keys_eog.detach().cpu().numpy()
                    # Add key-logit pair to the memory bank
                    memory_eeg.push(keys_eeg, eeg_output)
                    memory_eog.push(keys_eog, eog_output)

                    # metrics
                    true_label_eeg = true_label_eeg.cpu().numpy()
                    pseudo_label = pseudo_label.cpu().numpy()
                    label_all_subject['p_label']+=pseudo_label.tolist()
                    label_all_subject['t_label']+=true_label_eeg.tolist()
                    test_accuracy = accuracy_score(true_label_eeg, pseudo_label)
                    acc.append(test_accuracy)
                    test_f1 = f1_score(true_label_eeg, pseudo_label, average='weighted')
                    f1.append(test_f1)
                    test_prec=precision_score(true_label_eeg,pseudo_label, average='weighted')
                    prec.append(test_prec)
                    test_rec=recall_score(true_label_eeg, pseudo_label, average='weighted')
                    rec.append(test_rec)

                    if step == 1:
                        acc1.append(test_accuracy)
                        f1_1.append(test_f1)
                    elif step == steps:
                        acc2.append(test_accuracy)
                        f1_2.append(test_f1)
                if steps >= 2:
                    print(f"acc1:{acc1},acc2:{acc2},f1_1:{f1_1},f1_2{f1_2}\n")
        # Write accuracy and f1-score to the log file.
        with open(log_file, 'a') as f:
            f.write(f"Test Accuracy for Subject{test_num + 1}: {sum(acc) / len(acc) * 100:.2f}%\n")
            f.write(f"Test F1 Score for Subject{test_num + 1}: {sum(f1) / len(f1) * 100:.2f}%\n")
            f.write(f"Test Precision Score for Subject{test_num + 1}: {sum(prec) / len(prec) * 100:.2f}%\n")
            f.write(f"Test Recall Score for Subject{test_num + 1}: {sum(rec) / len(rec) * 100:.2f}%\n")
            f.write(f"Total Loss for Subject{test_num + 1}: {running_loss / len(eeg_test_loader)}\n")

        metrics_all_subject['acc'].append(sum(acc) / len(acc))
        metrics_all_subject['f1'].append(sum(f1) / len(f1))
        metrics_all_subject['prec'].append(sum(prec) / len(prec))
        metrics_all_subject['rec'].append(sum(rec) / len(rec))

        test_num += 1      

    # confusion matrix
    print(f"t_label:{len(label_all_subject['t_label'])}")
    print(f"Test Accuracy: {sum(metrics_all_subject['acc']) / len(metrics_all_subject['acc']) * 100:.2f}%\n")
    print(f"Test F1 Score: {sum(metrics_all_subject['f1']) / len(metrics_all_subject['f1']) * 100:.2f}%\n")
    print(f"Test Precision Score: {sum(metrics_all_subject['prec']) / len(metrics_all_subject['prec']) * 100:.2f}%\n")
    print(f"Test Recall Score: {sum(metrics_all_subject['rec']) / len(metrics_all_subject['rec']) * 100:.2f}%\n")
    print(f"subject size:{len(metrics_all_subject['acc'])}")

    ShowConfusionMatrix(label_all_subject['t_label'], label_all_subject['p_label'], savePath='./conf_matrix/SHHSto153', title=title_name)

def ShowConfusionMatrix(y_true, y_pred, classes=['W','N1','N2','N3','REM'], savePath='./conf_matrix/', title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix.
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n=cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]*100,'.2f')+'%\n'+format(cm_n[i, j],'d'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath+title+".png")
    plt.show()
    return ax


def SingleConfig(path,
                 device="cpu"):
    """Configure the model and optimizer."""
    # Modify the key in the pth file to match the model.
    path_to_pth_file = path
    checkpoint = torch.load(path_to_pth_file, map_location=device)

    new_checkpoint = {}
    # Iterate over the key-value pairs of the original dictionary and change the names of the keys
    for key, value in checkpoint.items():
        new_key = key.replace('ssn.', '')
        new_checkpoint[new_key] = value
    # Saving the new dictionary back to the original pth file.
    torch.save(new_checkpoint, path_to_pth_file)
    # Configure the model and optimizer.
    pretrained_model_params = torch.load(path)
    model_a = SleepTTA().to(device)
    model_a.load_state_dict(pretrained_model_params)
    configure_model(model_a, "fast")
    params_a, param_names = collect_params(model_a)
    optimizer_a = setup_optimizer(params_a)
    logger.info(f"model for adaptation: %s", model_a)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer_a)

    model_r = SleepTTA().to(device)
    model_r.load_state_dict(pretrained_model_params)
    configure_model(model_r, "slow")
    return model_a, optimizer_a, model_r


def setup_optimizer(params):
    """Configure the optimizer."""
    if optimizer_method == 'Adam':
        return optim.Adam(params,
                          lr=lr,
                          betas=(0.9, 0.999),
                          weight_decay=0.0)
    elif optimizer_method == 'SGD':
        return optim.SGD(params,
                         lr=lr,
                         momentum=0.9,
                         dampening=0.0,
                         weight_decay=0.0,
                         nesterov=True)
    else:
        raise NotImplementedError


def softmax_entropy(p1, p2, y):
    """Entropy of softmax distribution from logits."""
    y = y.long()
    return nn.CrossEntropyLoss()(p1, y) + nn.CrossEntropyLoss()(p2, y)


def collect_params(model):
    """
    Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_model(model, model_type="none"):
    """Configure model for adaptation by test-time normalization."""
    model.train()
    model.requires_grad_(False)
    # enable grad + force batch statisics
    if model_type == "fast":
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model


def data_split(dataset_list, total_num):
    """ Split the data. """
    dataset_list_total = []
    for i in range(0, total_num):
        cur_dataset_list = []
        cur_dataset_list.append(dataset_list[i])
        dataset_list_total.append(cur_dataset_list)
    return dataset_list_total


def sim(x, y, epsilon=1e-10):
    """Calculate the similarity metric."""
    kl_xy = F.kl_div(y.log(), x, reduction="sum")
    kl_yx = F.kl_div(x.log(), y, reduction="sum")
    return ((1 / (kl_xy + epsilon) + 1 / (kl_yx + epsilon)) / 2).item()


def balanced(sim_eeg, sim_eog, eeg_output, eog_output):
    """A balance model fusion stategy for merging pseudo-labels."""
    weight_eeg = sim_eeg / (sim_eog + sim_eeg)
    weight_eog = 1 - weight_eeg
    output = weight_eeg * eeg_output + weight_eog * eog_output
    _, pseudo_label = torch.max(output, dim=1)
    return pseudo_label


def similar(p_out, logits)->float:
    """Measure the similarity between the output of a modality and that of its neighbours from the memory bank."""
    p_out=p_out.cpu()
    logits=logits.cpu()
    kl=(F.kl_div(p_out.log(), logits, reduction="sum")+F.kl_div(logits.log(), p_out, reduction="sum"))*0.5
    kl=args.beta/kl
    return torch.sigmoid(kl).item()


if __name__ == '__main__':
    evaluate()
