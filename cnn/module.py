import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.quasirandom import SobolEngine
from scipy.stats.qmc import Halton


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, init_std, bias=True, device=None, dtype=None, qmc_method=None):
        """
        weight: (out_features, in_features)
        bias: (out_features,)
        input_buf: (bs, in_features)
        epsilon_buf: (bs, out_features)
        noise_std: (out_features,)
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_features,), np.log(init_std), device=device))
        self.input_buf = None
        self.epsilon_buf = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = SobolEngine(2 * out_features)
        elif self.qmc_method == 'halton':
            self.halton_engine = Halton(2 * out_features)

    def forward(self, input, add_noise=False):
        """
        input: (bs, in_features)
        logit_output: (bs, out_features)
        """
        logit_output = super().forward(input)
        if add_noise:
            bs, out_features = logit_output.shape
            epsilon = torch.zeros_like(logit_output, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                if self.qmc_method == 'sobol':
                    samples = self.sobol_engine.draw(bs//2)
                else:
                    samples = torch.from_numpy(self.halton_engine.random(bs // 2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :self.out_features] + 1e-8)) \
                                 * torch.cos(2 * np.pi * samples[:, self.out_features:])
                epsilon[:bs//2] += normal_samples.view(bs // 2, out_features).to(self.log_noise_std.device)
            else:
                epsilon[:bs//2] += torch.randn((bs//2, out_features), device=self.log_noise_std.device)
            epsilon[bs//2:] -= epsilon[:bs//2]

            noise = epsilon * torch.exp(self.log_noise_std)
            self.input_buf = input
            self.epsilon_buf = epsilon
            return logit_output + noise
        else:
            return logit_output

    def backward(self, loss):
        """
        loss: (bs,)
        """
        batch_size = self.input_buf.shape[0]
        loss = loss.unsqueeze(-1)
        noise_std = torch.exp(torch.unsqueeze(self.log_noise_std,-1))

        self.weight.grad = torch.einsum('ni,nj->ji', self.input_buf * loss, self.epsilon_buf) / (noise_std * batch_size)
        self.bias.grad = torch.einsum('ni,nj->j', loss, self.epsilon_buf) / (torch.exp(self.log_noise_std) * batch_size)
        self.log_noise_std.grad = torch.einsum('ni,nj->j', loss, self.epsilon_buf ** 2 - 1) / batch_size
        # self.log_noise_std.grad = None

        self.input_buf = None
        self.epsilon_buf = None

    def fetch_gradient(self):
        return self.weight.grad.detach().cpu()


class Linear_(nn.Linear):
    def __init__(self, in_features, out_features, init_std, bias=True, device=None, dtype=None, qmc_method=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_features,), np.log(init_std), device=device))
        self.epsilon_buf = None
        self.epsilon_buf_b = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = SobolEngine(2 * (self.weight.data.numel()+self.bias.data.numel()))
        elif self.qmc_method == 'halton':
            self.halton_engine = Halton(2 * (self.weight.data.numel()+self.bias.data.numel()))

    def forward(self, input, add_noise=False):
        if add_noise:
            bs = input.shape[0]
            w = self.weight.unsqueeze(0).repeat(bs,1,1)
            b = self.bias.unsqueeze(0).repeat(bs,1)
            epsilon_w = torch.zeros_like(w, device=self.log_noise_std.device)
            epsilon_b = torch.zeros_like(b, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                num_w, num_b = self.weight.data.numel(), self.bias.data.numel()
                if self.qmc_method == 'sobol':
                    samples = self.sobol_engine.draw(bs//2)
                else:
                    samples = torch.from_numpy(self.halton_engine.random(bs//2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :num_w + num_b] + 1e-12)) \
                                 * torch.cos(2 * np.pi * samples[:, num_w + num_b:])
                epsilon_w[:bs // 2] += normal_samples[:, :num_w].reshape(bs // 2, self.out_features, self.in_features).to(self.log_noise_std.device)
                epsilon_b[:bs // 2] += normal_samples[:, num_w:].reshape(bs // 2, self.out_features).to(self.log_noise_std.device)
            else:
                epsilon_w[:bs // 2] += torch.randn((bs // 2, self.out_features, self.in_features), device=self.log_noise_std.device)
                epsilon_b[:bs // 2] += torch.randn((bs // 2, self.out_features), device=self.log_noise_std.device)

            epsilon_w[bs // 2:] -= epsilon_w[:bs // 2]
            epsilon_b[bs // 2:] -= epsilon_b[:bs // 2]
            self.epsilon_buf = epsilon_w
            self.epsilon_buf_b = epsilon_b
            w += epsilon_w * torch.exp(self.log_noise_std[None, :, None])
            b += epsilon_b * torch.exp(self.log_noise_std[None, :])

            logit_output = torch.bmm(w,input[:,:,None]).squeeze(-1) + b
        else:
            logit_output = super().forward(input)
        return logit_output

    def backward(self, loss):
        bs = loss.shape[0]
        tmp = loss[:, None, None] * self.epsilon_buf
        self.weight.grad = torch.sum(tmp, 0) / (bs * torch.exp(self.log_noise_std[:, None]))

        tmp = loss[:, None] * self.epsilon_buf_b
        self.bias.grad = torch.sum(tmp, 0) / (bs * torch.exp(self.log_noise_std))

        tmp = torch.sum((self.epsilon_buf ** 2) - 1, dim=2) + ((self.epsilon_buf_b ** 2) - 1)
        self.log_noise_std.grad = torch.sum(tmp * loss[:, None], 0) / bs
        # self.log_noise_std.grad = None

        self.epsilon_buf = None
        self.epsilon_buf_b = None
    
    def fetch_gradient(self):
        return self.weight.grad.detach().cpu()


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, init_std,
                 bias=True, device=None, dtype=None, qmc_method=None):
        """
        weight: (out_channels, in_channels, H, W)
        bias: (out_channels,)
        input_buf: (N, in_channels, H, W)
        epsilon_buf: (N, out_channels, H_, W_)
        noise_std: (out_channels,)
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         bias=bias, device=device, dtype=dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_channels,), np.log(init_std), device=device))
        self.input_buf = None
        self.epsilon_buf = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = None
        elif self.qmc_method == 'halton':
            self.halton_engine = None

    def forward(self, input, add_noise=False):
        """
        input: (N, in_channels, H, W)
        logit_output: (N, out_channels, H_, W_)
        """
        logit_output = super().forward(input)
        if add_noise:
            N, out_channels, H_, W_ = logit_output.shape
            epsilon = torch.zeros_like(logit_output, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                if self.qmc_method == 'sobol':
                    if self.sobol_engine is None:
                        self.sobol_engine = SobolEngine(2 * out_channels * H_ * W_)
                    samples = self.sobol_engine.draw(N//2)
                else:
                    if self.halton_engine is None:
                        self.halton_engine = Halton(2 * out_channels * H_ * W_)
                    samples = torch.from_numpy(self.halton_engine.random(N//2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :out_channels * H_ * W_] + 1e-8)) \
                                 * torch.cos(2 * np.pi * samples[:, out_channels * H_ * W_:])
                epsilon[:N//2] += normal_samples.view(N // 2, out_channels, H_, W_).to(self.log_noise_std.device)
            else:
                epsilon[:N//2] += torch.randn((N//2, out_channels, H_, W_), device=self.log_noise_std.device)
            epsilon[N//2:] -= epsilon[:N//2]
            noise = epsilon * torch.exp(self.log_noise_std[None,:,None,None])
            self.input_buf = input
            self.epsilon_buf = epsilon
            return logit_output + noise
        else:
            return logit_output

    def backward(self, loss):
        """
        loss: (N,)
        """
        N, C_in, H_in, W_in = self.input_buf.shape
        _, C_out, H_out, W_out = self.epsilon_buf.shape

        # weight
        tmp = self.input_buf * loss[:,None,None,None]
        tmp = tmp.transpose(0,1)
        epsilon_buf = self.epsilon_buf.transpose(0,1)
        grad = F.conv2d(tmp,epsilon_buf,torch.zeros(size=(C_out,),device=self.log_noise_std.device),
                        dilation=self.stride, padding=self.padding)
        if grad.shape[2]>self.weight.shape[2]:
            grad = grad[:,:,:self.weight.shape[2],:]
        if grad.shape[3]>self.weight.shape[3]:
            grad = grad[:,:,:,:self.weight.shape[3]]
        self.weight.grad = grad.transpose(0,1) / (N * torch.exp(self.log_noise_std[:,None,None,None]))

        # bias
        tmp = torch.sum(self.epsilon_buf,(2,3)) * loss[:,None]
        self.bias.grad = torch.sum(tmp, 0) / (N * torch.exp(self.log_noise_std))

        # noise std
        tmp = torch.sum(self.epsilon_buf**2 - 1, (2,3)) * loss[:,None]
        self.log_noise_std.grad = torch.sum(tmp, 0) / N
        # self.log_noise_std.grad = None

        self.input_buf = None
        self.epsilon_buf = None

    def fetch_gradient(self):
        return self.weight.grad.detach().cpu()


class Conv2d_(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, init_std,
                 bias=True, device=None, dtype=None, qmc_method=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         bias=bias, device=device, dtype=dtype)
        self.log_noise_std = nn.Parameter(torch.full((out_channels,), np.log(init_std), device=device))
        self.epsilon_buf = None
        self.epsilon_buf_b = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = SobolEngine(2 * (self.weight.data.numel()+self.bias.data.numel()))
        elif self.qmc_method == 'halton':
            self.halton_engine = Halton(2 * (self.weight.data.numel()+self.bias.data.numel()))

    def forward(self, input, add_noise=False):
        if add_noise:
            C_out, C_in, H, W = self.weight.data.shape
            N, _, H_in, W_in = input.shape

            noise_std = torch.exp(self.log_noise_std).repeat(N)
            w = self.weight.repeat(N,1,1,1)
            b = self.bias.repeat(N)
            epsilon_w = torch.zeros_like(w, device=self.log_noise_std.device)
            epsilon_b = torch.zeros_like(b, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                num_w, num_b = self.weight.data.numel(), self.bias.data.numel()
                if self.qmc_method == 'sobol':
                    samples = self.sobol_engine.draw(N//2)
                else:
                    samples = torch.from_numpy(self.halton_engine.random(N//2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :num_w + num_b] + 1e-12)) \
                                 * torch.cos(2 * np.pi * samples[:, num_w + num_b:])
                epsilon_w[:N * C_out // 2] += normal_samples[:, :num_w].reshape(N * C_out // 2, C_in, H, W).to(self.log_noise_std.device)
                epsilon_b[:N * C_out // 2] += normal_samples[:, num_w:].reshape(N * C_out // 2).to(self.log_noise_std.device)
            else:
                epsilon_w[:N * C_out // 2] += torch.randn((N * C_out // 2, C_in, H, W), device=self.log_noise_std.device)
                epsilon_b[:N * C_out // 2] += torch.randn((N * C_out // 2,), device=self.log_noise_std.device)

            epsilon_w[N * C_out // 2:] -= epsilon_w[:N * C_out // 2]
            epsilon_b[N * C_out // 2:] -= epsilon_b[:N * C_out // 2]
            self.epsilon_buf = epsilon_w
            self.epsilon_buf_b = epsilon_b

            w += epsilon_w * noise_std[:, None, None, None]
            b += epsilon_b * noise_std

            logit_output = F.conv2d(input.reshape(1,N*C_in,H_in, W_in), w, b, stride=self.stride, padding=self.padding, groups=N)
            _, _, H_out, W_out = logit_output.shape
            logit_output = logit_output.reshape(N,C_out,H_out,W_out)
        else:
            logit_output = super().forward(input)
        return logit_output

    def backward(self, loss):
        N = loss.shape[0]
        # weight
        tmp_w = torch.stack(torch.split(self.epsilon_buf, split_size_or_sections=self.out_channels, dim=0))    # N, C_out, C_in, H, W
        tmp = loss[:,None,None,None,None] * tmp_w
        self.weight.grad = torch.sum(tmp, dim=0) / (N * torch.exp(self.log_noise_std[:,None,None,None]))

        # bias
        tmp_b = torch.stack(torch.split(self.epsilon_buf_b, split_size_or_sections=self.out_channels, dim=0))  # N, C_out
        tmp = loss[:, None] * tmp_b
        self.bias.grad = torch.sum(tmp, 0) / (N * torch.exp(self.log_noise_std))

        # noise std
        tmp = torch.sum((tmp_w**2) - 1, dim=[2, 3, 4]) + ((tmp_b**2) - 1)
        self.log_noise_std.grad = torch.sum(tmp * loss[:,None], 0) / N
        # self.log_noise_std.grad = None

        self.epsilon_buf = None
        self.epsilon_buf_b = None

    def fetch_gradient(self):
        return self.weight.grad.detach().cpu()


class BatchNorm2d(nn.BatchNorm2d):
    
    def __init__(self, num_features, init_std, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, qmc_method=None):
        """
        weight: (num_features,)
        bias: (num_features,)
        input_buf: (N, num_features, H, W)
        epsilon_buf: (N, num_features, H, W)
        noise_std: (num_features,)
        """
        super().__init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.log_noise_std = nn.Parameter(torch.full((num_features,), np.log(init_std), device=device))
        self.input_buf = None
        self.epsilon_buf = None
        self.qmc_method = qmc_method
        if self.qmc_method == 'sobol':
            self.sobol_engine = None
        elif self.qmc_method == 'halton':
            self.halton_engine = None
    
    def forward_(self, input, add_noise=False):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if add_noise:
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)
        else:
            bn_training = False

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return  F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
    
    def forward(self, input, add_noise=False):
        logit_output = super().forward(input)
        # logit_output = self.forward_(input, add_noise)
        
        if add_noise:
            N, out_channels, H_, W_ = logit_output.shape
            epsilon = torch.zeros_like(logit_output, device=self.log_noise_std.device)

            if self.qmc_method is not None:
                if self.qmc_method == 'sobol':
                    if self.sobol_engine is None:
                        self.sobol_engine = SobolEngine(2 * out_channels * H_ * W_)
                    samples = self.sobol_engine.draw(N//2)
                else:
                    if self.halton_engine is None:
                        self.halton_engine = Halton(2 * out_channels * H_ * W_)
                    samples = torch.from_numpy(self.halton_engine.random(N//2)).to(torch.float32)
                normal_samples = torch.sqrt(-2 * torch.log(samples[:, :out_channels * H_ * W_] + 1e-8)) \
                                 * torch.cos(2 * np.pi * samples[:, out_channels * H_ * W_:])
                epsilon[:N//2] += normal_samples.view(N // 2, out_channels, H_, W_).to(self.log_noise_std.device)
            else:
                epsilon[:N//2] += torch.randn((N//2, out_channels, H_, W_), device=self.log_noise_std.device)
            epsilon[N//2:] -= epsilon[:N//2]
            noise = epsilon * torch.exp(self.log_noise_std[None,:,None,None])
            self.input_buf = input
            self.epsilon_buf = epsilon
            self.norm_input_buf = (input - self.running_mean[None,:,None,None]) / torch.sqrt(self.running_var[None,:,None,None] + self.eps)
            # self.norm_input_buf = F.batch_norm(
            #     input,
            #     # If buffers are not to be tracked, ensure that they won't be updated
            #     self.running_mean,
            #     self.running_var,
            #     self.weight,
            #     self.bias,
            #     False,
            #     self.momentum,
            #     self.eps,
            # )
            return logit_output + noise
        else:
            return logit_output
        
    def backward(self, loss):
        """
        loss: (N,)
        """
        N, C_in, H_in, W_in = self.norm_input_buf.shape
        _, C_out, H_out, W_out = self.epsilon_buf.shape

        # weight
        tmp = self.norm_input_buf * loss[:,None,None,None]
        epsilon_buf = self.epsilon_buf
        grad = torch.einsum('bchw,bchw->c', tmp, epsilon_buf)
        self.weight.grad = grad / (N * torch.exp(self.log_noise_std))

        # bias
        tmp = torch.sum(self.epsilon_buf,(2,3)) * loss[:,None]
        self.bias.grad = torch.sum(tmp, 0) / (N * torch.exp(self.log_noise_std))

        # noise std
        tmp = torch.sum(self.epsilon_buf**2 - 1, (2,3)) * loss[:,None]
        self.log_noise_std.grad = torch.sum(tmp, 0) / N
        # self.log_noise_std.grad = None

        self.input_buf = None
        self.epsilon_buf = None
        self.norm_input_buf = None

    def fetch_gradient(self):
        return self.weight.grad.detach().cpu()


class Sequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input, add_noise=False):
        for module in self:
            try:
                input = module(input, add_noise)
            except TypeError:
                input = module(input)
        return input

    def backward(self, loss):
        for module in self:
            try:
                if module.epsilon_buf is not None:
                    module.backward(loss)
            except AttributeError:
                continue

    def fetch_gradient(self):
        gradient_list = []
        for module in self:
            try:
                gradient_list.append(module.fetch_gradient())
            except AttributeError:
                continue
        if len(gradient_list)==1:
            return gradient_list[0]
        else:
            return gradient_list
        
        
class BasicBlock(nn.Module):
    
    def __init__(self, conv, in_channels, out_channels, kernel_size, ds_size, init_std,
                 bias=True, device=None, dtype=None, qmc_method=None, if_residual=True, norm_layer=None):
        super().__init__()
        self.ds_size = ds_size
        self.if_residual = if_residual
        
        if norm_layer == None:
            norm_layer = BatchNorm2d
        self.conv1 = conv(in_channels, out_channels, kernel_size, ds_size, tuple((np.array(kernel_size)-1)//2), init_std,
                            bias, device, dtype, qmc_method)
        self.bn1 = norm_layer(out_channels, init_std)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, kernel_size, 1, tuple((np.array(kernel_size)-1)//2), init_std,
                            bias, device, dtype, qmc_method)
        self.bn2 = norm_layer(out_channels, init_std)
        
        if self.if_residual:
            if in_channels != out_channels or (ds_size not in [(1,1), 1, None]):
                self.input_connect = Sequential(conv(in_channels, out_channels, (1,1), ds_size, 0, init_std),
                                                norm_layer(out_channels, init_std))
            else:
                self.input_connect = Sequential(nn.Identity())
        
        self.epsilon_buf = None

    def forward(self, input, add_noise=False):
        if add_noise:
            self.epsilon_buf = 1

        x = input

        x = self.conv1(x,add_noise)
        x = self.bn1(x,add_noise)
        x = self.relu(x)

        x = self.conv2(x,add_noise)
        x = self.bn2(x,add_noise)

        if self.if_residual:
            x += self.input_connect(input, add_noise)

        x = self.relu(x)

        return x

    def backward(self, loss):
        if self.conv1.epsilon_buf is not None:
                self.conv1.backward(loss)
        if self.bn1.epsilon_buf is not None:
                self.bn1.backward(loss)
        if self.conv2.epsilon_buf is not None:
                self.conv2.backward(loss)
        if self.bn2.epsilon_buf is not None:
                self.bn2.backward(loss)
        if self.if_residual:
            self.input_connect.backward(loss)
        self.epsilon_buf = None
    
    def fetch_gradient(self):
        gradient_list = []
        gradient_list.append(self.conv1.fetch_gradient())
        gradient_list.append(self.bn1.fetch_gradient())
        gradient_list.append(self.conv2.fetch_gradient())
        gradient_list.append(self.bn2.fetch_gradient())
        gradient_list.append(self.input_connect.fetch_gradient())
        if len(gradient_list)==1:
            return gradient_list[0]
        else:
            return gradient_list
        
        
class BasicBlock_wo_BN(nn.Module):
    
    def __init__(self, conv, in_channels, out_channels, kernel_size, ds_size, init_std,
                 bias=True, device=None, dtype=None, qmc_method=None, if_residual=True, norm_layer=None):
        super().__init__()
        self.ds_size = ds_size
        self.if_residual = if_residual
        
        if norm_layer == None:
            norm_layer = BatchNorm2d
        self.conv1 = conv(in_channels, out_channels, kernel_size, ds_size, tuple((np.array(kernel_size)-1)//2), init_std,
                            bias, device, dtype, qmc_method)
        # self.bn1 = norm_layer(out_channels, init_std)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, kernel_size, 1, tuple((np.array(kernel_size)-1)//2), init_std,
                            bias, device, dtype, qmc_method)
        # self.bn2 = norm_layer(out_channels, init_std)
        
        if self.if_residual:
            if in_channels != out_channels or (ds_size not in [(1,1), 1, None]):
                self.input_connect = Sequential(conv(in_channels, out_channels, (1,1), ds_size, 0, init_std))
            else:
                self.input_connect = Sequential(nn.Identity())
        
        self.epsilon_buf = None

    def forward(self, input, add_noise=False):
        if add_noise:
            self.epsilon_buf = 1
        x = input

        x = self.conv1(x,add_noise)
        # x = self.bn1(x,add_noise)
        x = self.relu(x)

        x = self.conv2(x,add_noise)
        # x = self.bn2(x,add_noise)

        if self.if_residual:
            x += self.input_connect(input, add_noise)

        x = self.relu(x)

        return x

    def backward(self, loss):
        if self.conv1.epsilon_buf is not None:
                self.conv1.backward(loss)
        # if self.bn1.epsilon_buf is not None:
        #         self.bn1.backward(loss)
        if self.conv2.epsilon_buf is not None:
                self.conv2.backward(loss)
        # if self.bn2.epsilon_buf is not None:
        #         self.bn2.backward(loss)
        if self.if_residual:
            self.input_connect.backward(loss)
        self.epsilon_buf = None
    
    def fetch_gradient(self):
        gradient_list = []
        gradient_list.append(self.conv1.fetch_gradient())
        # gradient_list.append(self.bn1.fetch_gradient())
        gradient_list.append(self.conv2.fetch_gradient())
        # gradient_list.append(self.bn2.fetch_gradient())
        gradient_list.append(self.input_connect.fetch_gradient())
        if len(gradient_list)==1:
            return gradient_list[0]
        else:
            return gradient_list
        
    