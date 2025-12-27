



# 为了兼容 train.py 中可能使用 model.module 的情况
# 创建一个简单的包装类，使 model.module 返回模型本身
class ModelWrapper:
    """
    单节点训练模型包装类，用于兼容 train.py 中的 model.module 引用
    
    这个包装类模拟了 DistributedDataParallel 的行为，使得单节点训练时
    代码可以像使用 DDP 一样访问 model.module，但实际上直接使用原始模型
    
    关键点：
    - self.module 直接指向原始模型，这样 model.module.xxx 可以正常工作
    - __getattr__ 转发所有属性访问到原始模型
    - 实现了 PyTorch 模型需要的主要方法
    """
    def __init__(self, model):
        # 关键：self.module 直接指向原始模型，这样 model.module.xxx 可以正常工作
        self.module = model  # 兼容 train.py 中的 model.module 访问
        self._model = model  # 保存原始模型引用
    
    def __call__(self, *args, **kwargs):
        """前向传播：直接调用原始模型"""
        return self._model(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        转发所有其他属性访问到原始模型
        
        这个方法会在 Python 找不到属性时被调用
        例如：model.visual, model.logit_scale 等都会通过这里转发到原始模型
        """
        # 避免递归：如果访问的是我们自己的属性，直接返回
        if name in ['module', '_model']:
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass
        
        # 转发到原始模型
        try:
            return getattr(self._model, name)
        except AttributeError:
            # 如果原始模型也没有这个属性，抛出清晰的错误
            raise AttributeError(f"'{type(self).__name__}' object and wrapped model both have no attribute '{name}'")
    
    def parameters(self):
        """返回模型参数"""
        return self._model.parameters()
    
    def named_parameters(self):
        """返回命名的模型参数"""
        return self._model.named_parameters()
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """返回模型状态字典"""
        return self._model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def load_state_dict(self, state_dict, strict=True):
        """加载模型状态字典"""
        return self._model.load_state_dict(state_dict, strict=strict)
    
    def train(self, mode=True):
        """设置模型为训练模式"""
        self._model.train(mode)
        return self
    
    def eval(self):
        """设置模型为评估模式"""
        self._model.eval()
        return self
    
    def modules(self):
        """返回所有模块（用于兼容 train.py 中的检查）"""
        return self._model.modules()
    
    def children(self):
        """返回直接子模块"""
        return self._model.children()
    
    def named_children(self):
        """返回命名的直接子模块"""
        return self._model.named_children()
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """返回命名的所有模块"""
        return self._model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
