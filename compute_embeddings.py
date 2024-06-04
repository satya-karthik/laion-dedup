import os
import argparse
from pathlib import Path
import sys
import torch
from feature_extractor import create_model
import torchvision
import numpy as np
import logging
try:
    import webdataset as wds
    has_wds = True
except ImportError:
    has_wds = False
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default="image_folder")
    parser.add_argument('--dataset_root', type=str, default="root")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_config', type=str, default="resnet50.th")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out_folder', type=str, default="out")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--out_format', type=str, default="npy")
    parser.add_argument('--batches_per_chunk', type=int, default=100)
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--verbose', default=False,
                        action="store_true", help="verbose mode")
    parser.add_argument('--distributed', default=False,
                        action="store_true", help="distributed mode")
    parser.add_argument('--dist_env', type=str, default="env://")
    parser.add_argument('--dist_backend', type=str, default="nccl")
    args = parser.parse_args()
    run(args)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def world_info_from_env():
    # from https://github.com/mlfoundations/open_clip/blob/1c8647f14ff1f826b9096962777e39f7c5cd4ba9/src/training/distributed.py
    # Thanks to OpenCLIP authors
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def get_dataloader(dataset_type: str, dataset_root: [str],
                   device: torch.device,
                   preprocessor: torchvision.transforms.transforms,
                   batch_size: int = 32, workers: int = 4
                   ) -> torch.utils.data.DataLoader:
    """Returns a dataloader for the specified dataset type and root directory.

    Args:
        dataset_type (str): The type of dataset to load. Can be "webdataset" or "image_folder".
        device (torch.device): The device on which to load the data.
        preprocessor (torchvision.transforms.transforms): A torchvision transform pipeline that will be applied to each image.
        dataset_root (list): The path to the root directory of the dataset.
        batch_size (int, optional): The batch size for the dataloader. Defaults to 32.
        workers (int, optional): The number of workers for the dataloader. Defaults to 4.

    Returns:
        torch.utils.data.DataLoader: A dataloader for the specified dataset.
    """
    if dataset_type == "webdataset":
        assert has_wds
        pipeline = [wds.SimpleShardList(dataset_root)]
        pipeline.extend([
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
        pipeline.extend([
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png"),
            wds.map_dict(image=preprocessor),
            wds.to_tuple("image", "json"),
            wds.batched(batch_size, partial=False),
        ])
        dataset = wds.DataPipeline(*pipeline)
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=workers,
        )
    elif dataset_type == "image_folder":
        dataset = torchvision.datasets.ImageFolder(
            dataset_root, transform=preprocessor)
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=workers, shuffle=False, batch_size=batch_size)
    else:
        raise ValueError(dataset_type)

    return dataloader


def run(args):
    if args.distributed:
        import utils
        utils.init_distributed_mode(args)
        local_rank, rank, world_size = world_info_from_env()
        print(local_rank, rank, world_size)
    else:
        rank = 0
        world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_folder = os.path.join(args.out_folder, "emb")
    meta_folder = os.path.join(args.out_folder, "meta")
    if rank == 0:
        os.makedirs(emb_folder, exist_ok=True)
        os.makedirs(meta_folder, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    weight_name = 'checkpoint_0009'
    model, preprocessor = create_model(
        weight_name=weight_name,
        device='cuda', model_dir="/weights")

    model = model.to(device)
    model.eval()
    args.dataset_root = map(
        str, Path(args.dataset_root).resolve().glob("*.tar"))

    for dataset_item in args.dataset_root:
        chuck_name = str(Path(dataset_item).resolve().stem)
        dataloader = get_dataloader(dataset_type="webdataset",
                                    dataset_root=[dataset_item], device=device,
                                    preprocessor=preprocessor)

        features_chunks = []
        meta_chunks = []

        for X, meta in tqdm(dataloader):

            file_meta = [{"tar_file": f"/data/{x['key'][:-4]}.tar",
                          "file_name": f"{x['key']}.jpg"} for x in meta]
            X = X.to(device)
            with torch.no_grad():
                features = model(X)
                features = features.data.cpu()
            features = features.view(len(features), -1)
            features = features.numpy()
            features_chunks.append(features)
            meta_chunks.extend(file_meta)

        features = np.concatenate(features_chunks)
        np.save(os.path.join(
            emb_folder, f"{chuck_name}"), features)
        np.save(os.path.join(meta_folder,
                f"{chuck_name}"), meta_chunks)


if __name__ == "__main__":
    sys.exit(main())
