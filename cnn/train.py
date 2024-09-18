import torch
import re, os
import datetime
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from model import ResNet5, VGG8, ResNet5_w_BN, ResNet18, ResNet18_wo_BN
from dataloader import get_train_valid_loader
import shutil
import random
import more_itertools


def save_model(state_dict, is_best, log_dir):
    torch.save(state_dict, log_dir+'/latest.pth')
    if is_best:
        torch.save(state_dict, log_dir+'/best.pth')


def save_py(log_dir, py_dir='./'):
    for filename in os.listdir(py_dir):
        if filename.endswith(".py"):
            src_path = os.path.join(py_dir, filename)
            dst_path = os.path.join(log_dir, 'codes', filename)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)


def gradient_cosine_similarity(x, y):
    assert len(x) == len(y)
    cosine_similarity = []
    for i in range(len(x)):
        # print(i)
        assert type(x[i]) == type(y[i])
        if type(x[i]) == list:
            cosine_similarity.append(gradient_cosine_similarity(x[i], y[i]))
        else:
            assert type(x[i]) == torch.Tensor
            cosine_similarity.append(float(torch.cosine_similarity(x[i].flatten(), y[i].flatten(), dim=0)))
    return cosine_similarity


def validation(net, valid_loader, criterion, device):
    valid_loss, valid_accuracy, valid_count = 0., 0., 0
    net.eval()
    for inputs, labels in tqdm(valid_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        valid_accuracy += torch.sum(torch.where(labels == torch.argmax(outputs, dim=1), 1, 0)).cpu().detach().numpy()
        valid_loss += torch.sum(loss).cpu().detach().numpy()
        valid_count += inputs.shape[0]
    net.train()
    valid_loss /= valid_count
    valid_accuracy /= valid_count
    return valid_loss, valid_accuracy


if __name__ == '__main__':
    is_lr = True    # use LR or BP
    is_compare = True  # compare the gradient with the other method

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device('cuda:0')

    # net = VGG8().to(device)
    # repeat_n = [100, 200, 400, 800, 400, 200, 100, 50]
    
    # net = ResNet5().to(device)
    # repeat_n = [800,800,800,800,100]
    
    # net = ResNet5_w_BN().to(device)
    # repeat_n = [100, 100, 200, 200, 200, 200, 100, 100, 50]
    
    # net = ResNet18().to(device)
    # repeat_n = [800, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100]
    
    net = ResNet18_wo_BN().to(device)
    repeat_n = [800, 400, 400, 400, 200, 200, 200, 200, 200, 100]
    
    print(type(net))
    
    batch_size = 100
    train_loader, valid_loader = get_train_valid_loader(data_dir='../data/cifar10', batch_size=batch_size)
    
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # log_folder_name = 'LR_0211215044' # ResNet18_wo_BN (good one)
    # log_folder_name = 'LR_0216124705' # ResNet18 (test one)
    # log_folder_name = 'LR_0216011603' # ResNet5
    # model_path = './logs/' + type(net).__name__ + f'/{log_folder_name}' + '/latest.pth'
    # optimizer_path = './logs/' + type(net).__name__ + f'/{log_folder_name}' + '/optimizer.pth'
    # scheduler_path = './logs/' + type(net).__name__ + f'/{log_folder_name}' + '/scheduler.pth'
    # net.load_state_dict(torch.load(model_path))
    # optimizer.load_state_dict(torch.load(optimizer_path))
    # scheduler.load_state_dict(torch.load(scheduler_path))
    # _, valid_accuracy = validation(net, valid_loader, criterion, device)
    # print(f'last epoch:{scheduler.last_epoch-1}, valid accuracy:{valid_accuracy*100:.2f}%')
    # net.set_sigma(0.01)
    # net.module_w_para[0].set_sigma(0.001)

    current_time = re.sub(r'\D', '', str(datetime.datetime.now())[4:-7])
    print(current_time)
    log_dir = './logs/'+ type(net).__name__ + '/LR_' + current_time if is_lr else \
        './logs/' + type(net).__name__ + '/BP_' + current_time
    writer = SummaryWriter(log_dir=log_dir)
    save_py(log_dir, py_dir='./cnn/')

    epochs = 100
    # epochs = 5

    net.train()
    best_accuracy = -1.
    layer_n = len(net.module_w_para)
    for epoch in range(scheduler.last_epoch, epochs):
        # is_lr = not is_lr
        if is_lr:
            # net.set_sigma(0.001)
            net.set_sigma(0.001+0.0001*(epoch))
            # net.module_w_para[0].set_sigma(0.001)
        train_loss, train_accuracy, train_count = 0., 0., 0
        if is_lr and is_compare:
            cosine_similarity_list = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, labels) in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if is_lr:
                outputs = net(inputs)
                loss_0 = criterion(outputs, labels)
                net.eval()
                if is_compare:
                    outputs = net(inputs)
                    loss_bp = criterion(outputs, labels)
                    loss_bp_ = torch.mean(loss_bp)
                    loss_bp_.backward()
                    grad_bp = net.fetch_gradient()
                    optimizer.zero_grad()
                with torch.no_grad():
                    # this part can be done on multi-GPUs parallely
                    for l in range(layer_n):
                        inputs_ = inputs.repeat(repeat_n[l], 1, 1, 1)
                        labels_ = labels.repeat(repeat_n[l])
                        add_noise = [True if j == l else False for j in range(layer_n)]
                        outputs = net(inputs_, add_noise)
                        loss = criterion(outputs, labels_)
                        delta_loss = loss - loss_0.repeat(repeat_n[l])
                        net.backward(delta_loss)
                        train_accuracy += torch.sum(torch.where(labels_ == torch.argmax(outputs, 1), 1, 0)).cpu().detach().numpy()
                        train_loss += torch.sum(loss).cpu().detach().numpy()
                        train_count += len(labels_)
                net.train()
                if is_compare:
                    grad_lr = net.fetch_gradient()
                    cosine_similarity_list.append(gradient_cosine_similarity(grad_bp, grad_lr))
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss_ = torch.mean(loss)
                loss_.backward()
                train_accuracy += torch.sum(torch.where(labels == torch.argmax(outputs, 1), 1, 0)).cpu().detach().numpy()
                train_loss += torch.sum(loss).cpu().detach().numpy()
                train_count += len(labels)
            optimizer.step()
            if is_lr and is_compare:
                mean_sim = np.mean([list(more_itertools.collapse(sim)) for sim in cosine_similarity_list],axis=0)
                pbar.set_description(f"grad_sim: {np.round(mean_sim[:5],4)}")
        scheduler.step()
        train_loss /= train_count
        train_accuracy /= train_count

        valid_loss, valid_accuracy = validation(net, valid_loader, criterion, device)
        print(f'Train Epoch:{epoch:3d} || train loss:{train_loss:.2e} train accuracy:{train_accuracy*100:.2f}% ' +
              f'valid loss:{valid_loss:.4e} valid accuracy:{valid_accuracy*100:.2f}% lr:{scheduler.get_last_lr()[0]:.2e}')
        if is_lr and is_compare:
            print(f"grad_sim: {np.round(mean_sim, 4)}")
        save_model(net.state_dict(), valid_accuracy >= best_accuracy, log_dir)
        torch.save(optimizer.state_dict(), log_dir + '/optimizer.pth')
        torch.save(scheduler.state_dict(), log_dir + '/scheduler.pth')
        best_accuracy = deepcopy(valid_accuracy) if valid_accuracy >= best_accuracy else best_accuracy
        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('loss/valid_loss', valid_loss, epoch)
        writer.add_scalar('accuracy/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('accuracy/valid_accuracy', valid_accuracy, epoch)

    print(f'Finished Training') 

# Resnet18 wo BN
# 1. update sigma using previous method: valid: 55.78
# 2. first_layer:conv2d_; incresing sigma: valid: 47.68 (epoch 89)
# 3. first_layer:conv2d; incresing sigma: valid: 61.56
# 4. continue learning from update with previous method; fisrt layer conv2d with sigma:0.001, other:0.1; valid: 64.03(epoch 77)(stop increasing)
# 5. same as 3., but first layer repeat 800: valid: 64.27
# 6. same as 5., but avgpool: valid: 67.88
# 7. same as 6., but first layer sigma always 0.001: valid: 44.98(epoch 11)(first layer similarity is worse than 6.)
# 8. same as 6., but loss - loss_0: valid: 67.19
# 9. same as 8., but ramdom seed 1,1,1: valid: 67.09
# 10. same as 8., but repeat [400, 400, 400, 400, 300, 300, 300, 300, 200, 100]: valid: 65.80
# 11. same as 8., but sigma increasing 0.0001: valid: 68.13 (epoch 99)
# 12. increase sigma 0.001 + 0.0001*(epoch), LR/BP alternate: valid: 58.53 (epoch 6) not promising
# 13. fixed a bug in input_connect add noise, minus loss_0, sigma 0.001 + 0.0001*(epoch), double repeat_n at epech 80: valid: 65.47 (epoch 79)
# 14. same as 11. but fixed the bug: valid: 68.75 (epoch 99)
# 15. same as 14. but one step bp in basicblock: valid: 65.51 (epoch 99)


# Resnet18
# 1. same as Resnet18 wo BN 7.: valid: overfitting
# 2. same as 1., but BN layer update running_mean and running_var only when add_noise is True: valid: (even worse than 1.)
# (probably because of the BN update the running_mean and running_var when it is adding noise, then the gradient before this BN and after are at different situation)
# 3. same as 1., but BN layer update running_mean and running_var before calculating all layers' gradient: valid: 34.43
# 4. same as Resnet18 wo BN 6.: valid: 39.37 (epoch 96)
# next step: try add_noise to all BNs first, then add_noise to each layer as usual

# Resnet5
# 1. first half is conv2d_, incresing sigma: valid: 60.48 (epoch 10)
# 2. first half is conv2d, incresing sigma: valid: 62.16 (epoch 68)
# 3. same as 2., but repeat [400,400,200,200,50]: valid: 48.53 (epoch 5)(similarity not very high)
# 4. same as 2., but init_sigma=0.01: valid: 31.06 (increasing then decreasing)(probably because of the sigma is too large)
# 5. same as 2., but withoud residual in the model: valid: normal, no difference
# 6. first half is conv2d, set sigma always 0.01: valid: 68.32
# 7. first half is conv2d, minus loss_0, increasing sigma 0.001, repeat [150,150,150,150,50]: valid: 66.08 (epech 48)
# 8. continue learning 7., set sigma always 0.001: valid: 69.72 (epoch 99) (LR_0216011603)
# 9. continue learning 8., but double repeat at epoch 100: valid: 70.74
# 10. increase sigma 0.001 + 0.0001*(epoch)., repeat [150,150,150,150,50], LR/BP alternate: valid: 72.76 (epoch 99)
# 11. minus loss_0, set sigma always 0.001, adam.beta1=0.95, double repeat_n at epech 80: valid: 70.39 (epoch 99)
# 12. minus loss_0, set sigma always 0.001, repeat [400,400,400,400,100]: valid: 72.47 (epoch 98)
# 13. minus loss_0, set sigma always 0.001, repeat [150,150,150,150,50], batch szie 500: valid: 68.45 (epoch 99)
# 14. same as 13. but batch size 200: valid: 69.32 (epoch 99)
# 15. same as 12. but set sigma always 0.0001: valid: 71.37 (epoch 99)
# 16. same as 12. repeat [800,800,800,800,100]: valid: 74.36 (epoch 99)
