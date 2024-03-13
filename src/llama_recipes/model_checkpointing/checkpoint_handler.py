# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import torch
import time
from concurrent.futures import ThreadPoolExecutor

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint._fsspec_filesystem import (
    FsspecWriter,
    FsspecReader,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg):
    # torch.manual_seed(103)
    model_basename = Path(cfg.model_name).name
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + model_basename
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
         print(f"loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")

def profile_async_writeout(f, rank, epoch):
    t_w = time.perf_counter()
    while not f.done():
        time.sleep(1)
        # if rank == 0:
        #     print(f"still waiting... {time.perf_counter() - t_w}")
    print(f"kinesis: Checkpoint writeout time (rank {rank}, epoch {epoch})... {time.perf_counter() - t_w}")

def save_model_and_optimizer_sharded(epoch, model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    chk_type = "async" #async or sync
    chk_writer = "fsspec" #filesystem or fsspec
    log_writeout = True
    model_basename = Path(cfg.model_name).name
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + model_basename
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        path=save_dir,
	thread_count=8,
	single_file_per_rank=False,
	sync_files=False
    )

#    # Only create temp_dir when rank is 0
#    if rank == 0:
#        temp_dir = tempfile.mkdtemp()
#        print(f"Using temp directory: {temp_dir}")
#    else:
#        temp_dir = ""
#    object_list = [temp_dir]
#
#    # Broadcast temp_dir to all the other ranks
#    dist.broadcast_object_list(object_list)
#    global_temp_dir = object_list[0]
#
#    fsspec_save_path = global_temp_dir
#    fsspec_writer = FsspecWriter(
#        fsspec_save_path
#    )

    fsspec_save_path = str(save_dir)
    fsspec_writer = FsspecWriter(
        path=fsspec_save_path,
	thread_count=8,
	single_file_per_rank=False,
	sync_files=False
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        t_state = time.perf_counter()
        state_dict = {"model": model.state_dict()}
        print(f"kinesis: Checkpoint state_dict creation time (rank {rank})... {time.perf_counter()-t_state}")
        
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)
            print(f"adding optim to state_dict")
        
        if (chk_writer == "fsspec"):
            print(f"Using fsspec_writer")
            str_writer = fsspec_writer
        else:
            print(f"Using distributed_writer")
            str_writer = distributed_writer

#            if (chk_type == "async"):
#                print(f"Doing async checkpointing with fsspec writer to {fsspec_save_path}")
#                f = dist_cp.state_dict_saver._async_save(
#                        state_dict=state_dict,
#                        #checkpoint_id=save_dir
#                        storage_writer=fsspec_writer,
#                        planner=DefaultSavePlanner(),
#                    
#                    )
#            else:    
#                print(f"Doing default checkpointing with fsspec writer to {fsspec_save_path}")
#                f = dist_cp.save_state_dict(
#                        state_dict=state_dict,
#                        #checkpoint_id=save_dir
#                        storage_writer=fsspec_writer,
#                        planner=DefaultSavePlanner(),
#                    
#                    )
##        t = time.monotonic()
##        while not f.done():
##            time.sleep(1)
##            print(f"still waiting... {time.monotonic() - t}")
##        f.result()
        if (chk_type == "async"):
            print(f"Doing async checkpointing to {save_dir}")
            t_m = time.perf_counter()
            f = dist_cp.state_dict_saver.async_save(
                state_dict=state_dict,
                storage_writer=str_writer,
                planner=DefaultSavePlanner(),
                
            )
            print(f"kinesis: Checkpoint memory copy time (rank {rank})... {time.perf_counter() - t_m}")
            
            if (log_writeout):
                executor = ThreadPoolExecutor(max_workers=1)
                executor.submit(
                    profile_async_writeout(f, rank, epoch),
                    f,
                    rank,
                    epoch,
                )
                # f.add_done_callback(lambda f: executor.shutdown(wait=False))
                executor.shutdown(wait=False)

        else:
            print(f"Doing sync checkpointing to {save_dir}")
            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=str_writer,
                planner=DefaultSavePlanner(),
                
            )
    t_b = time.perf_counter()
    dist.barrier()
    t1 = time.perf_counter()
    print(f"kinesis: Checkpoint barrier time = {t1-t_b:.4f}")
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(f"kinesis: Checkpoint Time = {t1-t0:.4f}")
def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        model_basename = Path(cfg.model_name).name
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + model_basename
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = model_basename + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")
      


def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return


    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    
    print(f"model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

    t0 = time.perf_counter()
    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    print(f"kinesis: Optim state dict creation time (rank {rank})... {time.perf_counter()-t0}")
    
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        model_basename = Path(cfg.model_name).name
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + model_basename
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + model_basename + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")
        t_save = time.perf_counter()
        torch.save(optim_state, opt_save_full_path)
        print(f"kinesis: Time to save optim state (rank {rank})... {time.perf_counter()-t_save}")

        print(f"--> saved {opt_save_full_path} to disk")
        print(f"kinesis: Time for save_optimizer_checkpoint (rank {rank})... {time.perf_counter()-t0}")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """


    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    print(f"optimizer shard loaded on rank {rank}")

def load_sharded_model_single_gpu(model,model_path):
    
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )
    
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model
