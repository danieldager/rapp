"""Token dataset for training language models.

Token Dataset uses WebDataset to stream tokenized audio data from .tar files.
It yields fixed-size blocks of tokens for training. Token sequences are packed.

"""

import os
import random
import tarfile
import traceback
from pathlib import Path
import webdataset as wds

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, IterableDataset


# CONSTANTS
SEED = 101
MAX_TOKENS = 2048
BOS_TOKEN_ID = 256
EOS_TOKEN_ID = 257
VOCAB_SIZE = 258  # 256 tokens + BOS + EOS
SHUFFLE_BUFFER = 1000


class TokenDataset(IterableDataset):
    def __init__(self, tokens_dir: str):

        # self.urls is all the paths strings to *.tar files in tokens_dir
        self.urls = sorted([str(p) for p in Path(tokens_dir).glob("*.tar")])

        self.block_size = MAX_TOKENS
        self.shuffle_buffer = SHUFFLE_BUFFER

        random.seed(SEED)
        random.shuffle(self.urls)

    def __iter__(self):
        dataset = (
            wds.WebDataset(  # type: ignore
                self.urls,
                resampled=True,
                shardshuffle=False,  # TODO: check this
                nodesplitter=wds.shardlists.split_by_node,
                handler=wds.warn_and_continue,  # type: ignore
            )
            .shuffle(self.shuffle_buffer)
            .decode()
        )
        dataset = dataset.compose(wds.shardlists.split_by_worker)

        buffer = []

        for sample in dataset:
            tokens = sample.get("tokens.npy")

            if tokens is None:
                print("Warning: 'tokens.npy' not found in sample, skipping.")
                continue  # Skip samples without tokens

            # token_tensor = torch.from_numpy(tokens).long()

            token_list = [BOS_TOKEN_ID] + tokens.tolist() + [EOS_TOKEN_ID]
            buffer.extend(token_list)

            # when we have enough tokens, cut and yield a block
            while len(buffer) >= self.block_size:
                block = buffer[: self.block_size]
                buffer = buffer[self.block_size :]

                yield {"input_ids": torch.tensor(block, dtype=torch.long)}


class EvalDataset(Dataset):
    def __init__(self, tokens_dir: str, num_blocks: int = 3000):

        self.urls = sorted([str(p) for p in Path(tokens_dir).glob("*.tar")])
        self.block_size = MAX_TOKENS
        self.num_blocks = num_blocks

        random.seed(SEED)
        random.shuffle(self.urls)

        dataset = wds.WebDataset(  # type: ignore
            self.urls,
            resampled=True,
            shardshuffle=False,  # TODO: check this
            nodesplitter=wds.shardlists.split_by_node,
            handler=wds.warn_and_continue,  # type: ignore
        ).decode()
        # dataset = dataset.compose(wds.shardlists.split_by_worker)

        buffer = []
        self.blocks = []

        for _, sample in enumerate(dataset):

            if len(self.blocks) < self.num_blocks:
                tokens = sample.get("tokens.npy")
                if tokens is None:
                    print("Warning: 'tokens.npy' not found in sample, skipping.")
                    continue  # Skip samples without tokens

                token_list = [BOS_TOKEN_ID] + tokens.tolist() + [EOS_TOKEN_ID]
                buffer.extend(token_list)

                # when we have enough tokens, cut and yield a block
                while len(buffer) >= self.block_size:
                    block = buffer[: self.block_size]
                    buffer = buffer[self.block_size :]

                    block_tensor = torch.tensor(block, dtype=torch.long)
                    self.blocks.append(block_tensor)

            else:
                break  # Stop if we've collected enough blocks

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        return {"input_ids": self.blocks[idx]}


def collate_fn(batch):
    tensors = torch.stack([item["input_ids"] for item in batch])  # [B, MAX_TOKENS]
    attention_mask = torch.ones_like(tensors)

    return {
        "input_ids": tensors,
        "labels": tensors,
        "attention_mask": attention_mask,
    }


def check_shards(tokens_dir):
    """Scans all .tar files in the directory for corruption."""
    print(f"\n--- Scanning for corrupted shards in {tokens_dir} ---")
    paths = sorted(list(Path(tokens_dir).glob("*.tar")))
    corrupted = []

    for p in paths:
        try:
            with tarfile.open(p, "r") as tar:
                # We don't need to extract everything, just try to list members
                # This catches 'empty header' or 'truncated file' errors
                _ = tar.getnames()
        except Exception as e:
            print(f"[CORRUPT] {p.name}: {e}")
            corrupted.append(p)

    print(f"Scan complete. {len(corrupted)}/{len(paths)} corrupted.")

    return corrupted


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    tokens_dir = "/scratch2/ddager/amela/tokens/chunks5_mhubert/"
    BATCH_SIZE = 4  # Small batch size for testing

    if rank == 0:
        bad_files = check_shards(tokens_dir)
        if bad_files:
            # You can choose to exit here or continue
            print("Proceeding with caution...\n")

    # if torch.cuda.is_available():
    #     torch.cuda.set_device(local_rank)
    #     dist.init_process_group(backend="nccl", init_method="env://")
    #     device = torch.device(f"cuda:{local_rank}")
    # else:
    #     device = torch.device("cpu")
    #     print("WARNING: CUDA not available, running on CPU")

    # print(f"[Rank {rank}] Initialized (Local Rank: {local_rank}, World: {world_size})")

    try:
        dataset = TokenDataset(tokens_dir)

        # We use a simple DataLoader.
        # Note: In DDP, we do NOT need a DistributedSampler because WebDataset
        # handles splitting via .split_by_node() in __iter__
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2,  # Testing with workers to ensure multiprocessing works
            pin_memory=True,
        )

        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            mask = batch["attention_mask"]

            # Stop after 1 batch for this test
            break

    except Exception as e:
        print(f"[Rank {rank}] CRITICAL ERROR: {e}")
        traceback.print_exc()

    finally:
        # Cleanup DDP
        if dist.is_initialized():
            dist.destroy_process_group()
