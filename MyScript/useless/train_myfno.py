import sys
import torch
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from pathlib import Path
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

from unit_dataset import UnitDataset
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training import setup, AdamW
from neuralop.utils import count_model_params, get_wandb_api_key
from neuralop.mpu.comm import get_local_rank

# ----------- 读取配置文件 -----------
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./my_config.yaml", config_name="default", config_folder="./config"
        ),
        ArgparseConfig(infer_types=True),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()

# ---------- 设置设备和分布式 ----------
device, is_logger = setup(config)

# ---------- 初始化 WandB ----------
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    wandb_name = (
        config.wandb.name
        or f"{config.fno.n_layers}_layer_{config.fno.hidden_channels}_hidden"
    )
    wandb.init(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )

# ----------- 打印配置 ------------
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

# ---------- 加载训练数据 ----------
train_dataset = UnitDataset(config.data.train_path)
train_loader = DataLoader(
    train_dataset, batch_size=config.data.batch_size, shuffle=True
)

# ---------- 加载多分辨率测试集 ----------
test_loaders = {}
for res, path, bs in zip(
    config.data.test_resolutions, config.data.test_paths, config.data.test_batch_sizes
):
    test_dataset = UnitDataset(path)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    test_loaders[res] = test_loader

# ---------- 构建模型 ----------
model = get_model(config).to(device)

# ---------- Patch 处理器 ----------
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(
        model=model,
        in_normalizer=None,
        out_normalizer=None,
        padding_fraction=config.patching.padding,
        stitching=config.patching.stitching,
        levels=config.patching.levels,
        use_distributed=config.distributed.use_distributed,
        device=device,
    )
else:
    data_processor = torch.nn.Identity()

# ---------- 设置分布式 Sampler ----------
if config.distributed.use_distributed:
    sampler = DistributedSampler(train_loader.dataset, rank=get_local_rank())
    train_loader = DataLoader(
        train_loader.dataset, batch_size=config.data.batch_size, sampler=sampler
    )

# ---------- 优化器 & 学习率调度 ----------
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Unknown scheduler {config.opt.scheduler}")

# ---------- 设置损失函数 ----------
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = l2loss if config.opt.training_loss == "l2" else h1loss
eval_losses = {"l2": l2loss, "h1": h1loss}

# ---------- 初始化训练器 ----------
trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_processor,
    mixed_precision=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    eval_interval=config.wandb.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
)

# ---------- 记录模型参数数 ----------
if is_logger:
    n_params = count_model_params(model)
    print(f"模型参数量: {n_params}")
    if config.wandb.log:
        wandb.log({"n_params": n_params})
        wandb.watch(model)

# ---------- 启动训练 ----------
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()
