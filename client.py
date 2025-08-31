import torch.optim as optim
from utils import *
from customopacus import PrivacyEngine
# from opacus import PrivacyEngine
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
import time
import copy
from collections import defaultdict


class DeCflClient(object):
    def __init__(self, id_num, train_d, val_d, device, train_config):
        self.id = id_num  # 客户端编号
        self.train_loader = train_d
        self.val_loader = val_d
        self.data_size = len(self.train_loader.dataset)
        self.device = device
        self.lr_mode = train_config.get('lr_mode')
        self.batch_size = train_config.get('batch_size')
        self.first_local_steps = 200
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.lr_decay = train_config.get('lr_decay')
        self.noise_decay = train_config.get('noise_decay')

        self.dW = None
        self.model_params = None
        self.criterion = nn.CrossEntropyLoss()
        self.privacy_engine = PrivacyEngine(accountant='prv')

    def len(self):
        return len(self.train_loader.dataset)

    def evaluate(self, model):
        device = self.device
        criterion = self.criterion.to(device)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train_py_aware(self, model, vk=None):
        model.train()
        vec_init = tensor_dict_to_vector(model.state_dict())
        train_loader = self.train_loader
        device = self.device
        criterion = self.criterion.to(device)
        noise_mul = self.noise_multiplier
        clip = self.grad_clip
        max_step = max(self.first_local_steps, self.local_steps)
        lr = self.lr
        if self.lr_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.lr_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        if noise_mul > 0:
            model, optimizer, train_loader = self.privacy_engine.make_private(
                Vk=vk,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=clip,
                poisson_sampling=False,
            )
        # 客户端训练
        step = 0
        bias = []
        bias_grad = []
        res = []
        loss = []
        while step < max_step:
            for data, target in train_loader:
                # print(step, optimizer.state_dict()['param_groups'][0]['lr'])
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                res.append(np.mean(torch.softmax(output, dim=-1).detach().cpu().numpy(), axis=0))
                t_loss = criterion(output, target.long())
                t_loss.backward()
                loss.append(t_loss.item())
                optimizer.step()  # 优化器更新语句后模型梯度才会被更新
                if noise_mul > 0 and optimizer.noise_multiplier > 0.5:
                    optimizer.noise_multiplier *= self.noise_decay
                if optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * self.lr_decay
                # 参数记录
                for k, v in model.named_parameters():
                    if noise_mul > 0 and k == '_module.fc.bias':
                        bias.append(v.detach().cpu().clone())
                        bias_grad.append(v.grad.clone().detach().cpu().clone())
                    if k == 'fc.bias':
                        bias.append(v.detach().cpu().clone())
                        bias_grad.append(v.grad.clone().detach().cpu().clone())
                step += 1
                if step >= max_step:
                    break
        # res = np.array(res)
        # loss = np.array(loss)
        # end_time = time.time()
        # v_loss, v_acc = self.evaluate(model)
        # print('Val acc:', v_acc)
        # print(f'id{self.id} Step{max_step} [v_loss={v_loss}, v_acc={v_acc}, time={end_time-start_time}]')
        # label_test(model, self.val_dataset, 'client' + str(self.id))
        # 计算模型总变化量
        local_models_params = tensor_dict_to_vector(model.state_dict())
        self.dW = local_models_params - vec_init
        self.model_params = model.state_dict()

        # 分析分类偏置层历史梯度
        # self.draw(bias_grad)
        # window = 5
        # M = []
        # for ii in range(max_step - window):  # 滑动窗搜索
        #     M.append(np.mean(bias_grad[ii:ii + window], axis=0))
        # M = np.array(M)
        # M = np.abs(M)
        # T = np.sort(np.max(M, axis=1))[::-1][int(max_step * 0.2)]
        # index = np.where(np.max(M, axis=1) < T)[0]
        # if index.size > 0:
        #     index = index[0]
        # else:
        #     raise ValueError('loss is too large, try increasing local_steps')

        index = 5
        data = torch.stack(bias_grad).numpy()
        bias_grad_std = np.std(data[index: index + 30], axis=0)
        pre_py = bias_grad_std / np.sum(bias_grad_std)

        true_py = np.bincount(train_loader.dataset.dataset.labels, minlength=len(pre_py))
        true_py = true_py / np.sum(true_py)

        py_dis = jensenshannon(true_py, pre_py)
        # print('JS(true,pre): ', py_dis)
        # print(np.sort(bias_grad_std))
        return pre_py, py_dis

    def train(self, model, vk=None):
        model.train()
        vec_init = tensor_dict_to_vector(model.state_dict())
        train_loader = self.train_loader
        device = self.device
        criterion = self.criterion.to(device)
        noise_mul = self.noise_multiplier
        clip = self.grad_clip
        max_step = self.local_steps
        lr = self.lr
        if self.lr_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.lr_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        if noise_mul > 0:
            model, optimizer, train_loader = self.privacy_engine.make_private(
                Vk=vk,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=clip,
                poisson_sampling=False,
            )
        step = 0
        while step < max_step:
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                t_loss = criterion(output, target.long())
                t_loss.backward()
                optimizer.step()  # 优化器更新语句后模型梯度才会被更新
                if noise_mul > 0 and optimizer.noise_multiplier > 0.5:
                    optimizer.noise_multiplier *= self.noise_decay
                if optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.lr = optimizer.param_groups[0]['lr']
                step += 1
                if step >= max_step:
                    break
        local_models_params = tensor_dict_to_vector(model.state_dict())
        self.dW = local_models_params - vec_init
        self.model_params = model.state_dict()


class FedAvgClient(object):
    def __init__(self, id_num, train_d, val_d, device, train_config):
        self.id = id_num  # 客户端编号
        self.train_loader = train_d
        self.val_loader = val_d
        self.data_size = len(self.train_loader.dataset)
        self.device = device
        self.lr_mode = train_config.get('lr_mode')
        self.batch_size = train_config.get('batch_size')
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.lr_decay = train_config.get('lr_decay')

        self.dW = None
        self.model_params = None
        self.criterion = nn.CrossEntropyLoss()

        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.privacy_engine = PrivacyEngine(accountant='prv')

    def len(self):
        return len(self.train_loader.dataset)

    def evaluate(self, model):
        device = self.device
        criterion = self.criterion.to(device)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, model, Vk=None):
        model.train()
        vec_init = tensor_dict_to_vector(model.state_dict())
        train_loader = self.train_loader
        device = self.device
        criterion = self.criterion.to(device)
        noise_mul = self.noise_multiplier
        clip = self.grad_clip
        max_step = self.local_steps
        lr = self.lr
        if self.lr_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.lr_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        # 客户端训练
        if noise_mul > 0:
            model, optimizer, train_loader = self.privacy_engine.make_private(
                Vk=Vk,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=clip,
                poisson_sampling=False,
            )
        step = 0
        while step < max_step:
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                t_loss = criterion(output, target.long())
                t_loss.backward()
                optimizer.step()  # 优化器更新语句后模型梯度才会被更新
                if optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.lr = optimizer.param_groups[0]['lr']
                step += 1
                if step >= max_step:
                    break
        # 计算模型总变化量
        local_models_params = tensor_dict_to_vector(model.state_dict())
        self.dW = local_models_params - vec_init
        self.model_params = model.state_dict()


class FedProxClient(object):
    def __init__(self, id_num, train_d, val_d, device, train_config, mu=0.1):
        self.id = id_num  # 客户端编号
        self.train_loader = train_d
        self.val_loader = val_d
        self.data_size = len(self.train_loader.dataset)
        self.device = device
        self.lr_mode = train_config.get('lr_mode')
        self.batch_size = train_config.get('batch_size')
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.lr_decay = train_config.get('lr_decay')

        self.dW = None
        self.model_params = None
        self.criterion = nn.CrossEntropyLoss()

        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.privacy_engine = PrivacyEngine(accountant='prv')

        self.mu = mu

    def len(self):
        return len(self.train_loader.dataset)

    def evaluate(self, model):
        device = self.device
        criterion = self.criterion.to(device)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, model):
        init_model = copy.deepcopy(model)
        model.train()
        vec_init = tensor_dict_to_vector(model.state_dict())
        train_loader = self.train_loader
        device = self.device
        criterion = self.criterion.to(device)
        noise_mul = self.noise_multiplier
        clip = self.grad_clip
        max_step = self.local_steps
        lr = self.lr
        if self.lr_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.lr_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        # 客户端训练
        if noise_mul > 0:
            model, optimizer, train_loader = self.privacy_engine.make_private(
                Vk=None,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=clip,
                poisson_sampling=False,
            )
        step = 0
        while step < max_step:
            for data, target in train_loader:
                # print(step, optimizer.state_dict()['param_groups'][0]['lr'])
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                t_loss = criterion(output, target.long())

                # 添加近端项
                proximal_term = 0.0
                for param, global_param in zip(
                    model.parameters(), init_model.parameters()
                ):
                    proximal_term += (param - global_param).norm(2)
                t_loss += (self.mu / 2) * proximal_term

                t_loss.backward()
                optimizer.step()  # 优化器更新语句后模型梯度才会被更新
                if optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.lr = optimizer.param_groups[0]['lr']
                step += 1
                if step >= max_step:
                    break
        # 计算模型总变化量
        local_models_params = tensor_dict_to_vector(model.state_dict())
        self.dW = local_models_params - vec_init
        self.model_params = model.state_dict()


class ScaffoldClient(object):
    def __init__(self, id_num, train_d, val_d, device, train_config, model):
        self.id = id_num  # 客户端编号
        self.train_loader = train_d
        self.val_loader = val_d
        self.data_size = len(self.train_loader.dataset)
        self.device = device
        self.lr_mode = train_config.get('lr_mode')
        self.batch_size = train_config.get('batch_size')
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.lr_decay = train_config.get('lr_decay')

        self.dW = None
        self.model_params = None
        self.criterion = nn.CrossEntropyLoss()

        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.privacy_engine = PrivacyEngine(accountant='prv')

        self.client_control = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        self.delta_control = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    def len(self):
        return len(self.train_loader.dataset)

    def evaluate(self, model):
        device = self.device
        criterion = self.criterion.to(device)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, model, server_control):
        model.train()
        vec_init = tensor_dict_to_vector(model.state_dict())
        train_loader = self.train_loader
        device = self.device
        criterion = self.criterion.to(device)
        noise_mul = self.noise_multiplier
        clip = self.grad_clip
        max_step = self.local_steps
        lr = self.lr
        if self.lr_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.lr_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        # 客户端训练
        if noise_mul > 0:
            model, optimizer, train_loader = self.privacy_engine.make_private(
                Vk=None,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=clip,
                poisson_sampling=False,
            )
        step = 0
        while step < max_step:
            for data, target in train_loader:
                # print(step, optimizer.state_dict()['param_groups'][0]['lr'])
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                t_loss = criterion(output, target.long())
                t_loss.backward()

                # 计算修正后的梯度
                for name, param in model.named_parameters():
                    if self.noise_multiplier > 0:
                        name =  name.split('_module.')[-1]
                    if param.grad is not None:
                        param.grad += (server_control[name] - self.client_control[name])

                optimizer.step()  # 优化器更新语句后模型梯度才会被更新
                if optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.lr = optimizer.param_groups[0]['lr']
                step += 1
                if step >= max_step:
                    break
        # 计算模型总变化量
        local_models_params = tensor_dict_to_vector(model.state_dict())
        self.dW = local_models_params - vec_init
        self.model_params = model.state_dict()

        # Update local control variate
        for name, param in model.named_parameters():
            if self.noise_multiplier > 0:
                name = name.split('_module.')[-1]
            self.delta_control[name] = self.client_control[name] - server_control[name] - (param.grad / max_step)
            self.client_control[name] = server_control[name] + self.delta_control[name]


class FlexCflClient(object):
    def __init__(self, id_num, train_d, val_d, device, train_config):
        self.id = id_num  # 客户端编号
        self.train_loader = train_d
        self.val_loader = val_d
        self.data_size = len(self.train_loader.dataset)
        self.device = device
        self.lr_mode = train_config.get('lr_mode')
        self.batch_size = train_config.get('batch_size')
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.lr_decay = train_config.get('lr_decay')

        self.dW = None
        self.model_params = None
        self.criterion = nn.CrossEntropyLoss()

        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.privacy_engine = PrivacyEngine(accountant='prv')

    def len(self):
        return len(self.train_loader.dataset)

    def evaluate(self, model):
        device = self.device
        criterion = self.criterion.to(device)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, model):
        model.train()
        vec_init = tensor_dict_to_vector(model.state_dict())
        train_loader = self.train_loader
        device = self.device
        criterion = self.criterion.to(device)
        noise_mul = self.noise_multiplier
        clip = self.grad_clip
        max_step = self.local_steps
        lr = self.lr
        if self.lr_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.lr_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        # 客户端训练
        if noise_mul > 0:
            model, optimizer, train_loader = self.privacy_engine.make_private(
                Vk=None,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=clip,
                poisson_sampling=False,
            )
        step = 0
        while step < max_step:
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                t_loss = criterion(output, target.long())
                t_loss.backward()
                optimizer.step()  # 优化器更新语句后模型梯度才会被更新
                if optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.lr = optimizer.param_groups[0]['lr']
                step += 1
                if step >= max_step:
                    break
        # 计算模型总变化量
        local_models_params = tensor_dict_to_vector(model.state_dict())
        self.dW = local_models_params - vec_init
        self.model_params = model.state_dict()


class FeSemClient(object):
    def __init__(self, id_num, train_d, val_d, device, train_config):
        self.id = id_num  # 客户端编号
        self.train_loader = train_d
        self.val_loader = val_d
        self.data_size = len(self.train_loader.dataset)
        self.device = device
        self.lr_mode = train_config.get('lr_mode')
        self.batch_size = train_config.get('batch_size')
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.lr_decay = train_config.get('lr_decay')

        self.dW = None
        self.model_params = None
        self.criterion = nn.CrossEntropyLoss()

        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.privacy_engine = PrivacyEngine(accountant='prv')

    def len(self):
        return len(self.train_loader.dataset)

    def evaluate(self, model):
        device = self.device
        criterion = self.criterion.to(device)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def train(self, model):
        model.train()
        vec_init = tensor_dict_to_vector(model.state_dict())
        train_loader = self.train_loader
        device = self.device
        criterion = self.criterion.to(device)
        max_step = self.local_steps
        noise_mul = self.noise_multiplier
        clip = self.grad_clip
        lr = self.lr
        if self.lr_mode == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif self.lr_mode == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        # 客户端训练
        if noise_mul > 0:
            model, optimizer, train_loader = self.privacy_engine.make_private(
                Vk=None,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_mul,
                max_grad_norm=clip,
                poisson_sampling=False,
            )
        step = 0
        while step < max_step:
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                t_loss = criterion(output, target.long())
                t_loss.backward()
                optimizer.step()  # 优化器更新语句后模型梯度才会被更新
                if optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * self.lr_decay
                    self.lr = optimizer.param_groups[0]['lr']
                step += 1
                if step >= max_step:
                    break
        # 计算模型总变化量
        local_models_params = tensor_dict_to_vector(model.state_dict())
        self.dW = local_models_params - vec_init
        self.model_params = model.state_dict()

from typing import List, Tuple, Dict, Iterable, Optional
import torch.nn.functional as F

class FedRCClientLite(object):
    """
    子进程内使用的精简客户端，避免pickle主进程的复杂对象。
    仅依赖：train_loader / val_loader / device / train_config
    """
    def __init__(self, train_loader, val_loader, device, train_config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.lr_decay = train_config.get('lr_decay')
        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.privacy_engine = PrivacyEngine(accountant='prv')
        self.criterion = nn.CrossEntropyLoss()

    def len(self):
        return len(self.train_loader.dataset)

    @torch.no_grad()
    def estimate_gamma_omega_for_client(self, models: List[nn.Module], num_classes: int = 10, max_val_batches: int = 10):
        device = self.device
        K = len(models)
        omega = torch.full((K,), 1.0 / K, device=device)
        C_yk = torch.ones(K, num_classes, device=device)
        gamma_sums = torch.zeros(K, device=device)
        sample_count = 0

        gammas_batches: List[torch.Tensor] = []
        batches = 0
        for batch in self.val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            B = x.size(0)
            logits_list = [models[k](x) for k in range(K)]
            losses_per_k = [F.cross_entropy(logits_list[k], y, reduction='none') for k in range(K)]
            losses = torch.stack(losses_per_k, dim=1)  # [B,K]
            exp_neg_loss = torch.exp(-losses)
            C_per_sample = torch.stack([C_yk[:, y[b]] for b in range(B)], dim=0)
            q = omega.view(1, K) * exp_neg_loss / C_per_sample
            gamma = q / (q.sum(dim=1, keepdim=True) + 1e-12)
            gammas_batches.append(gamma.detach())
            for b in range(B):
                C_yk[:, y[b]] += gamma[b]
            gamma_sums += gamma.sum(dim=0)
            sample_count += B
            batches += 1
            if max_val_batches is not None and batches >= max_val_batches:
                break

        if sample_count > 0:
            omega = gamma_sums / sample_count
            omega = omega / (omega.sum() + 1e-12)

        return gammas_batches, omega

    def state_dict_sub(self, a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        res = {}
        for k in a.keys():
            if self.noise_multiplier > 0:
                group_k = k.split('_module.')[-1]
            else:
                group_k = k
            res[group_k] = a[k] - b[group_k]
        return res

    def local_weighted_update(self, global_models: List[nn.Module], gammas_batches: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        train_loader = self.train_loader
        device = self.device
        local_epochs = self.local_steps
        lr = self.lr
        momentum = 0.9
        weight_decay = 0
        grad_clip = self.grad_clip
        noise_multiplier = self.noise_multiplier
        privacy_engine = self.privacy_engine

        K = len(global_models)
        local_models = [copy.deepcopy(global_models[k]).to(device).train() for k in range(K)]
        optimizers = [
            torch.optim.SGD(local_models[k].parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            for k in range(K)
        ]

        # 仅包装一次
        if noise_multiplier > 0:
            for k in range(K):
                local_models[k], optimizers[k], train_loader = privacy_engine.make_private(
                    Vk=None,
                    module=local_models[k],
                    optimizer=optimizers[k],
                    data_loader=train_loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=grad_clip,
                    poisson_sampling=False,
                )
        batch_idx = 0
        while batch_idx < self.local_steps:
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                gamma = gammas_batches[min(batch_idx, len(gammas_batches)-1)].to(device)
                for k in range(K):
                    optimizers[k].zero_grad(set_to_none=True)
                    logits = local_models[k](x)
                    ce = F.cross_entropy(logits, y, reduction='none')
                    weights = gamma[:, k]
                    denom = weights.sum() + 1e-12
                    loss = torch.sum(weights * ce) / denom
                    loss.backward()
                    optimizers[k].step()
                    if optimizers[k].param_groups[0]['lr'] > 0.01:
                        optimizers[k].param_groups[0]['lr'] = optimizers[k].param_groups[0]['lr'] * self.lr_decay
                self.lr =optimizers[0].param_groups[0]['lr']
                batch_idx += 1
                if batch_idx >= self.local_steps:
                    break

        deltas = []
        for k in range(K):
            new_sd = local_models[k].state_dict()
            old_sd = global_models[k].state_dict()
            # 参数名一致
            delta = self.state_dict_sub(new_sd, old_sd)
            # delta = {kk: (new_sd[kk] - old_sd[kk]).detach().cpu() for kk in new_sd.keys()}
            deltas.append(delta)
        return deltas