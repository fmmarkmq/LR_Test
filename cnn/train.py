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


def save_py(log_dir):
    for filename in os.listdir('./'):
        if filename.endswith(".py"):
            src_path = os.path.join('./', filename)
            dst_path = os.path.join(log_dir, filename)
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
    # repeat_n = [100, 200, 200, 100, 50]
    
    # net = ResNet5_w_BN().to(device)
    # repeat_n = [100, 100, 200, 200, 200, 200, 100, 100, 50]
    
    # net = ResNet18().to(device)
    # repeat_n = [100, 200, 200, 200, 200, 400, 400, 400, 400, 100]
    
    net = ResNet18_wo_BN().to(device)
    repeat_n = [100, 200, 200, 400, 400, 400, 400, 200, 200, 100]
    
    print(type(net))
    
    current_time = re.sub(r'\D', '', str(datetime.datetime.now())[4:-7])
    log_dir = './logs/'+ type(net).__name__ + '/LR_' + current_time if is_lr else \
        './logs/' + type(net).__name__ + '/BP_' + current_time
    writer = SummaryWriter(log_dir=log_dir)
    save_py(log_dir)
    
    batch_size = 100
    train_loader, valid_loader = get_train_valid_loader(data_dir='../data/cifar10', batch_size=batch_size)

    epochs = 100
    # epochs = 5
    
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    best_accuracy = -1.
    layer_n = len(net.module_w_para)
    for epoch in range(epochs):
        train_loss, train_accuracy, train_count = 0., 0., 0
        if is_compare:
            cosine_similarity_list = []
        pbar = tqdm(train_loader, total=len(train_loader))
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if is_lr:
                if is_compare:
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss_ = torch.mean(loss)
                    loss_.backward()
                    grad_bp = net.fetch_gradient()
                    optimizer.zero_grad()
                with torch.no_grad():
                    # this part can be done on multi-GPUs parallely
                    for i in range(layer_n):
                        inputs_ = inputs.repeat(repeat_n[i], 1, 1, 1)
                        labels_ = labels.repeat(repeat_n[i])
                        add_noise = [True if j == i else False for j in range(layer_n)]
                        outputs = net(inputs_, add_noise)
                        loss = criterion(outputs, labels_)
                        net.backward(loss)
                        train_accuracy += torch.sum(torch.where(labels_ == torch.argmax(outputs, 1), 1, 0)).cpu().detach().numpy()
                        train_loss += torch.sum(loss).cpu().detach().numpy()
                        train_count += len(labels_)
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
            if is_compare:
                mean_sim = np.mean([list(more_itertools.collapse(sim)) for sim in cosine_similarity_list],axis=0)
                pbar.set_description(f"grad_sim: {np.round(mean_sim[-5:],4)}")         
        scheduler.step()
        train_loss /= train_count
        train_accuracy /= train_count

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
        torch.exp(net.module_w_para[0][0].log_noise_std)
        print(f'Train Epoch:{epoch:3d} || train loss:{train_loss:.2e} train accuracy:{train_accuracy*100:.2f}% ' +
              f'valid loss:{valid_loss:.4e} valid accuracy:{valid_accuracy*100:.2f}% lr:{scheduler.get_last_lr()[0]:.2e}')
        if is_compare:
            print(f"grad_sim: {np.round(mean_sim, 4)}")
            print(torch.exp(net.module_w_para[0][0].log_noise_std))
            print(torch.exp(net.module_w_para[4][0].conv1.log_noise_std))
            print(torch.exp(net.module_w_para[-2][0].conv1.log_noise_std))
            import pdb; pdb.set_trace()
        save_model(net.state_dict(), valid_accuracy >= best_accuracy, log_dir)
        torch.save(optimizer.state_dict(), log_dir + '/optimizer.pth')
        torch.save(scheduler.state_dict(), log_dir + '/scheduler.pth')
        best_accuracy = deepcopy(valid_accuracy) if valid_accuracy >= best_accuracy else best_accuracy
        writer.add_scalar('loss/train_loss', train_loss, epoch)
        writer.add_scalar('loss/valid_loss', valid_loss, epoch)
        writer.add_scalar('accuracy/train_accuracy', train_accuracy, epoch)
        writer.add_scalar('accuracy/valid_accuracy', valid_accuracy, epoch)

    print(f'Finished Training') 

# not update sigma
# epoch45: grad_sim: [0.0733 0.1417 0.1484 0.1149 0.0863 0.0984 0.1134 0.1964 0.1212 0.089
#  0.1279 0.1341 0.146  0.1506 0.0801 0.0782 0.0719 0.0768 0.0993 0.0907
#  0.9251]