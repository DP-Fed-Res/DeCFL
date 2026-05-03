import gc
import copy
import itertools
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from customopacus import PrivacyEngine
# from opacus import PrivacyEngine
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any, Optional


class BaseClient(object):
    """
    基础联邦学习客户端（基类）
    """

    def __init__(self, id_num, train_d, train_config):
        self.id = id_num  # 客户端编号
        self.train_loader = train_d
        self.data_size = len(self.train_loader.dataset)

        # 训练配置
        self.device = train_config.get('device')
        self.lr_mode = train_config.get('lr_mode')
        self.freeze_feature = train_config.get('freeze_1st')
        self.local_steps = train_config.get('tc')
        self.first_local_steps = train_config.get('tc_1st')
        self.lr = train_config.get('lr')
        self.clip = train_config.get('clip')
        self.noise_mul = train_config.get('noise')
        self.momentum = 0.9

        self.privacy_engine = PrivacyEngine(accountant='prv')
        self.criterion = nn.CrossEntropyLoss()

        self.model_params = None
        self.params_dir = f'./saved_models/client_{id_num}'
        os.makedirs(self.params_dir, exist_ok=True)

    def __len__(self):
        return self.data_size

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        if self.lr_mode == 'SGD':
            return optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.lr_mode == 'AdamW':
            return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.lr_mode}")

    def _freeze_feature_layer(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            raise AttributeError("Model does not have 'fc' attribute")

    def _apply_privacy(self, model: nn.Module, optimizer: optim.Optimizer, vk: Optional[Any] = None):
        model, optimizer, train_loader = self.privacy_engine.make_private(
            Vk=vk,
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_mul,
            max_grad_norm=self.clip,
            poisson_sampling=False,
        )
        return model, optimizer, train_loader

    def _save_model(self, model: nn.Module, r: int) -> None:
        save_path = os.path.join(self.params_dir, f'round_{r}.pt')
        torch.save(model.cpu().state_dict(), save_path)

    def get_infinite_batches(self, data_loader):
        """构建一个无限循环的数据生成器"""
        while True:
            for data, target in data_loader:
                yield data, target

    def train(self, model: nn.Module, r: int, vk: Optional[Any], **kwargs) -> None:
        try:
            model.train()
            max_step = self.first_local_steps if r == 0 else self.local_steps

            # 创建优化器
            optimizer = self._create_optimizer(model)

            # 第一轮冻结首层
            if self.freeze_feature and r == 0:
                self._freeze_feature_layer(model)

            # 应用隐私保护
            if self.noise_mul > 0:
                model, optimizer, train_loader = self._apply_privacy(model, optimizer, vk)
            else:
                train_loader = self.train_loader

            # 训练循环
            batch_generator = iter(self.get_infinite_batches(train_loader))
            for step in range(max_step):
                data, target = next(batch_generator)

                # data已在dataloader中处理
                target = target.long().to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                # if (step+1) % 50 == 0:
                #     pred = output.argmax(dim=1, keepdim=True)
                #     accuracy = 100. * pred.eq(target.view_as(pred)).sum().item() / len(target)
                #     print(f'client {self.id} training step {step+1} | Acc : {accuracy:.2f}%')

            # 维护模型参数
            self.model_params = {k: v.cpu() for k, v in model.state_dict().items()}

        except Exception as e:
            raise Exception(f"Training failed at round {r}: {str(e)}")


class FedProxClient(BaseClient):
    def __init__(self, id_num, train_d, train_config, mu):
        super().__init__(id_num, train_d, train_config)
        self.mu = mu

    def train(self, model: nn.Module, r: int, vk: Optional[Any], **kwargs) -> None:
        try:
            model.train()

            max_step = self.first_local_steps if r == 0 else self.local_steps

            # 创建优化器
            optimizer = self._create_optimizer(model)

            # 第一轮冻结特征层
            if self.freeze_feature and r == 0:
                self._freeze_feature_layer(model)

            # 应用隐私保护
            if self.noise_mul > 0:
                model, optimizer, train_loader = self._apply_privacy(model, optimizer, vk)
            else:
                train_loader = self.train_loader

            init_model = copy.deepcopy(model)

            # 训练循环
            batch_generator = iter(self.get_infinite_batches(train_loader))
            for step in range(max_step):
                # 每次无脑抽取下一个批次的数据，如果跑完一轮 dataloader，生成器会自动无缝重头开始
                data, target = next(batch_generator)

                # data已在dataloader中处理
                target = target.long().to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)

                # 添加近端项
                proximal_term = 0.0
                init_params_dict = dict(init_model.named_parameters())
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if name in init_params_dict:
                            init_param = init_params_dict[name]
                            proximal_term += (param - init_param.to(param.device)).norm(2) ** 2

                loss += (self.mu / 2) * proximal_term
                # print(f'noise_mul: {self.noise_mul}, mu: {self.mu}, proximal_term: {proximal_term:.4f}, CEloss: {loss - (self.mu / 2) * proximal_term:.4f}')

                loss.backward()
                optimizer.step()

                # if (step+1) % 50 == 0:
                #     pred = output.argmax(dim=1, keepdim=True)
                #     accuracy = 100. * pred.eq(target.view_as(pred)).sum().item() / len(target)
                #     print(f'client {self.id} training step {step+1} | Acc : {accuracy:.2f}%')

            # 维护模型参数
            self.model_params = {k: v.cpu() for k, v in model.state_dict().items()}

        except Exception as e:
            raise Exception(f"Training failed at round {r}: {str(e)}")


class ScaffoldClient(BaseClient):
    def __init__(self, id_num, train_d, train_config, model):
        super().__init__(id_num, train_d, train_config)
        self.param_names = [name.replace('_module.', '') for name, _ in model.named_parameters()]

        # 初始化控制变量（直接使用预处理后的名称）
        self.client_control = {
            clean_name: torch.zeros_like(param)
            for clean_name, (_, param) in zip(self.param_names, model.named_parameters())
        }
        self.delta_control = {
            name: torch.zeros_like(param)
            for name, param in self.client_control.items()
        }

    def _update_control_vars(self, initial_weights, final_weights, server_control, max_step):
        """更新控制变量（核心Scaffold算法）"""
        for name in self.param_names:
            dy = final_weights[name] - initial_weights[name]
            c_server_val = server_control[name]
            c_client_val = self.client_control[name]

            # 计算新控制变量: c_i_tilde = c_i_old - c_server - (Delta y)/(K*lr)
            c_i_tilde = c_client_val - c_server_val - (dy / (max_step * self.lr))

            # 更新差值和当前控制变量
            self.delta_control[name] = c_i_tilde - c_client_val
            self.client_control[name] = c_i_tilde

    def train(self, model, r, vk, **kwargs):
        server_control: Dict = kwargs.get('server_control', {})

        try:
            model.train()

            initial_weights = {name: param.detach().clone() for name, param in model.named_parameters()}

            max_step = self.first_local_steps if r == 0 else self.local_steps

            # 创建优化器
            optimizer = self._create_optimizer(model)

            # 第一轮冻结特征层
            if self.freeze_feature and r == 0:
                self._freeze_feature_layer(model)

            # 应用隐私保护
            if self.noise_mul > 0:
                model, optimizer, train_loader = self._apply_privacy(model, optimizer, vk)
            else:
                train_loader = self.train_loader

            # 训练循环
            batch_generator = iter(self.get_infinite_batches(train_loader))
            for step in range(max_step):
                # 每次无脑抽取下一个批次的数据，如果跑完一轮 dataloader，生成器会自动无缝重头开始
                data, target = next(batch_generator)

                # data已在dataloader中处理
                target = target.long().to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        # 处理 Opacus 自动添加的 '_module.' 前缀
                        clean_name = name.replace('_module.', '')

                        c_server_val = server_control[clean_name].to(self.device)
                        c_client_val = self.client_control[clean_name].to(self.device)

                        # 直接修正参数权重
                        param.data.add_(self.lr * (c_client_val - c_server_val))

                # if (step+1) % 10 == 0:
                #     pred = output.argmax(dim=1, keepdim=True)
                #     accuracy = 100. * pred.eq(target.view_as(pred)).sum().item() / len(target)
                #     print(f'client {self.id} training step {step+1} | Acc : {accuracy:.2f}%')

            final_weights = {
                name: param.detach().clone()
                for name, param in zip(self.param_names, model.parameters())
            }

            self._update_control_vars(initial_weights, final_weights, server_control, max_step)

            # 维护模型参数
            self.model_params = {k: v.cpu() for k, v in model.state_dict().items()}

        except Exception as e:
            raise Exception(f"Training failed at round {r}: {str(e)}")


class FlexCflClient(BaseClient):
    def __init__(self, id_num, train_d, train_config):
        super().__init__(id_num, train_d, train_config)


class FeSemClient(BaseClient):
    def __init__(self, id_num, train_d, train_config):
        super().__init__(id_num, train_d, train_config)


class FedRCClient(BaseClient):
    def __init__(self, id_num, train_d, train_config, K):
        super().__init__(id_num, train_d, train_config)
        # FedRC 特有的温度参数，用于控制软聚类的平滑程度 (对应论文中的 lambda)
        self.temperature = 1.0
        self.num_clusters = K
        self.pi_weights = None

    def train(self, model: List[nn.Module], r: int, vk: Optional[Any], **kwargs) -> None:
        """
        FedRC 客户端训练逻辑
        :param global_state_dicts: 包含 K 个聚类中心模型 state_dict 的列表
        :return: (K个更新后的 state_dict 列表, 客户端对这K个模型的分配权重 pi)
        """
        K = len(model)
        models = [copy.deepcopy(m).to(self.device) for m in model]
        for m in models:
            m.eval()

        # 2. 计算 FedRC 软分配权重 (Soft Assignments, \pi)
        losses = torch.zeros(K).to(self.device)
        with torch.no_grad():
            for x, y in self.train_loader:
                y = y.to(self.device)
                for k in range(K):
                    out = models[k](x)
                    loss = self.criterion(out, y)
                    losses[k] += loss.item()

        # 对损失取负并除以温度参数，再用 Softmax 转换为概率权重
        losses = losses / len(self.train_loader)
        pi_weights = F.softmax(-losses / self.temperature, dim=0)

        # 3. 本地执行 K 个模型的更新 (M-step 近似)
        updated_state_dicts = []
        for k in range(K):
            try:
                model = models[k]
                model.train()
                max_step = self.first_local_steps if r == 0 else self.local_steps

                # 创建优化器
                optimizer = self._create_optimizer(model)

                # 第一轮冻结首层
                if self.freeze_feature and r == 0:
                    self._freeze_feature_layer(model)

                # 应用隐私保护
                if self.noise_mul > 0:
                    model, optimizer, train_loader = self._apply_privacy(model, optimizer, vk)
                else:
                    train_loader = self.train_loader

                # 训练循环
                batch_generator = iter(self.get_infinite_batches(train_loader))
                for step in range(max_step):
                    data, target = next(batch_generator)

                    # data已在dataloader中处理
                    target = target.long().to(self.device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    # if (step+1) % 50 == 0:
                    #     pred = output.argmax(dim=1, keepdim=True)
                    #     accuracy = 100. * pred.eq(target.view_as(pred)).sum().item() / len(target)
                    #     print(f'client {self.id} training step {step+1} | Acc : {accuracy:.2f}%')

                # 维护模型参数
                updated_state_dicts.append({k.replace('_module.', ''): v.cpu() for k, v in model.state_dict().items()})

            except Exception as e:
                raise Exception(f"Training failed at round {r}: {str(e)}")

        self.model_params = updated_state_dicts
        self.pi_weights = pi_weights.cpu().numpy()


class DeCflClient(BaseClient):
    """
    DeCFL 客户端（子类），继承自 FedAvgClient
    包含数据增强多样性、DistilBERT 优化器分组及 Py-aware 训练逻辑
    """

    def __init__(self, id_num, train_d, train_config):
        super().__init__(id_num, train_d, train_config)

    def train(self, model: nn.Module, r: int, vk: Optional[Any], **kwargs):
        try:
            model.train()
            max_step = self.first_local_steps if r == 0 else self.local_steps

            # 创建优化器
            optimizer = self._create_optimizer(model)

            # 第一轮冻结特征层
            if self.freeze_feature and r == 0:
                self._freeze_feature_layer(model)

            # 应用隐私保护
            if self.noise_mul > 0:
                model, optimizer, train_loader = self._apply_privacy(model, optimizer, vk)
            else:
                train_loader = self.train_loader

            # 训练循环
            bias_grad = []
            batch_generator = iter(self.get_infinite_batches(train_loader))
            for step in range(max_step):
                data, target = next(batch_generator)

                # data已在dataloader中处理
                target = target.long().to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                if r == 0:
                    for k, v in model.named_parameters():
                        if (self.noise_mul > 0 and k == '_module.fc.bias') or (k == 'fc.bias'):
                            bias_grad.append(v.grad.clone().detach().cpu().clone())

                # if (step+1) % 10 == 0:
                #     pred = output.argmax(dim=1, keepdim=True)
                #     accuracy = 100. * pred.eq(target.view_as(pred)).sum().item() / len(target)
                #     print(f'client {self.id} training step {step+1} | Acc : {accuracy:.2f}%')

            # 维护模型参数
            self.model_params = model.cpu().state_dict()

            if r == 0:
                st = 20
                ed = min(st + 30, max_step)
                data = torch.stack(bias_grad).numpy()
                bias_grad_std = np.std(data[st: ed], axis=0)
                pre_py = bias_grad_std / np.sum(bias_grad_std)

                true_py = np.bincount(self.train_loader.dataset.labels, minlength=len(pre_py))
                true_py = true_py / np.sum(true_py)
                py_dis = jensenshannon(true_py, pre_py)

                return pre_py, py_dis

        except Exception as e:
            raise Exception(f"Training failed at round {r}: {str(e)}")
