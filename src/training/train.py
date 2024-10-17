import os
from os.path import exists
import time
import torch
from torch.optim.lr_scheduler import LambdaLR
import GPUtil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from architecture import EncoderDecoder, make_model
from training import TrainState, Batch, rate, SpecialTokens, SupportedDatasets, SupportedLanguages, TrainConfig, LabelSmoothing, SimpleLossCompute, DummyOptimizer, DummyScheduler, Preprocessor, create_dataloaders

def run_epoch(
    data_iter,
    model: EncoderDecoder,
    loss_compute: SimpleLossCompute,
    optimizer:torch.optim.Adam,
    scheduler:LambdaLR,
    mode:str="train",
    accum_iter:int=1,
    train_state:TrainState=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
        
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

def train_worker(
    gpu: int,
    ngpus_per_node: int,
    config: TrainConfig,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)
    
    preprocessor = Preprocessor(config.source_language,config.target_language,config.max_padding)

    pad_idx = preprocessor.vocab_tgt[SpecialTokens.blank.value]
    d_model = 512
    model = make_model(preprocessor.vocab_src_size, preprocessor.vocab_tgt_size, N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=preprocessor.vocab_tgt_size, padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        preprocessor,
        config.dataset,
        batch_size=config.batch_size // ngpus_per_node,
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config.warmup
        ),
    )
    train_state = TrainState()

    for epoch in range(config.num_epochs):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config.accum_iter,
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = f"{config.file_prefix}{epoch}.pt"
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = f"{config.file_prefix}final.pt"
        torch.save(module.state_dict(), file_path)
        
def train_distributed_model(config):
    from train import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, config, True),
    )

def train_model(
    dataset:str=SupportedDatasets.multi30k.value,
    source_language:str=SupportedLanguages.german.value,
    target_language:str=SupportedLanguages.english.value,
    batch_size:int=32,
    distributed:bool=False,
    num_epochs:int=8,
    accum_iter:int=10,
    base_lr:float=1.0,
    max_padding:int=72,
    warmup:int=3000,
    file_prefix:str="multi30k_model_",
):
    config = TrainConfig(
        dataset=dataset,
        source_language=source_language,
        target_language=target_language,
        batch_size=batch_size,
        distributed=distributed,
        num_epochs=num_epochs,
        accum_iter=accum_iter,
        base_lr=base_lr,
        max_padding=max_padding,
        warmup=warmup,
        file_prefix=file_prefix,
    )

    if distributed:
        train_distributed_model(config)
    else:
        train_worker(0, 1, config, False)

if __name__ == "__main__":
    train_model()