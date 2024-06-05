from glob import glob
from clize import run
import os
from pathlib import Path
import torch
import faiss
import numpy as np
import joblib
from feature_extractor import create_model
import webdataset as wds
from compute_embeddings import log_and_continue
from tqdm import tqdm
import sqlite3
from multiprocessing import Process, Queue
from typing import List, Dict


class MetaData:

    def __init__(self, folder):
        self.folder = folder

    def build(self):
        self.paths = list(sorted(glob(os.path.join(self.folder, "*.npy"))))
        self.sizes = []
        for path in self.paths:
            data = np.load(path, allow_pickle=True)
            self.sizes.append(len(data))

    def get_indices(self, indices):
        return [self.get(ind) for ind in indices]

    def get(self, index):
        start = 0
        for path, size in zip(self.paths, self.sizes):
            end = start + size
            if start <= index < end:
                data = np.load(path, allow_pickle=True)
                offset = index - start
                return data[offset]
            start = end


def process_images(L: np.ndarray, D: np.ndarray,
                   I_: np.ndarray, meta_features: List[Dict],
                   metadata: MetaData, db_name: str) -> None:
    """Process images and store the similarities in a SQLite database.

    Parameters
    ----------
    L : np.ndarray
        Array of indices.
    D : np.ndarray
        Array of distances.
    I_ : np.ndarray
        Array of image identifiers.
    meta_features : List[Dict]
        List of metadata features for each image.
    metadata : MetaData
        Metadata object containing additional information about the images.
    db_name : str
        Name of the SQLite database file to store the results in.

    Returns
    -------
    None
        This function does not return a value, it only performs side effects (modifying the SQLite database).
    """

    # Connect to the database
    with sqlite3.connect(db_name) as db:
        cursor = db.cursor()

        try:
            # Create table if not present
            query = '''CREATE TABLE IF NOT EXISTS image_similarities (ID INTEGER PRIMARY KEY AUTOINCREMENT, QueryImageName TEXT, SimilarImageName TEXT, SimilarityScore FLOAT)'''
            cursor.execute(query)
        except sqlite3.Error as e:
            print('Error while creating the table', e)
            return

        for i in range(len(L) - 1):
            indices = I_[L[i]: L[i + 1]]
            dists = D[L[i]: L[i + 1]]
            if len(indices) > 1:
                order = np.argsort(-dists)
                # removing the first index as it is always the same as the query image
                indices = indices[order][1:]
                dists = dists[order][1:]
                # extracting the meta for query image
                file_meta = meta_features[0][i]
                query_image = f"{file_meta['tar_file']}/{file_meta['file_name']}"
                # get all the similar images.
                meta_files = metadata.get_indices(indices)
                # change the meta_files into strings
                similar_images = [
                    f"{file_meta['tar_file']}/{file_meta['file_name']}" for file_meta in meta_files]
            else:
                # extracting the meta for query image
                file_meta = meta_features[0][i]
                query_image = f"{file_meta['tar_file']}/{file_meta['file_name']}"
                similar_images = ["None"]
                dists = [0]
            # Add entries to table
            for similar_image, score in zip(similar_images, dists):
                query = f'''INSERT INTO image_similarities (QueryImageName, SimilarImageName, SimilarityScore) VALUES ("{query_image}", "{similar_image}", {score})'''
                try:
                    cursor.execute(query)
                except sqlite3.Error as e:
                    print('Error while inserting to the table', e)
        # committing changes only after done with all the images.
        db.commit()
        cursor.close()


def worker(q):
    while True:
        # Get a task from the queue
        item = q.get()
        if item is None:  # Use a sentinel value to indicate end of tasks
            break
        # these things will change based on the task being processed and function used
        L, D, I_, meta_features, metadata, db_name = item
        process_images(L=L, D=D, I_=I_,
                       meta_features=meta_features,
                       metadata=metadata,
                       db_name=db_name)


@torch.no_grad()
def main(
    *,
    dataset_name="Laion_tar_2",
    root="root",
    cosine_similarity_threshold=0.604169,
    split="test",
    index="/output/index/knn.index",
    metadata="/output/meta",
    batch_size=128,
    num_workers=4,
    model_config="dedup_seer1.5B.th",
    out="/output/results/exp_010_t_0_604169",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset_slug = dataset_name.replace("/", "_")

    # Load metadata and index
    print("Loading metadata and index...")
    if os.path.isdir(metadata):
        metadata = MetaData(metadata)
        metadata.build()
        joblib.dump(metadata, os.path.join(os.path.dirname(index), "meta.pkl"))
    else:
        metadata = joblib.load(metadata)

    # metadata.cumsum = np.cumsum(metadata.sizes)
    start = 0
    metadata.path_index = []
    for i, size in enumerate(metadata.sizes):
        end = start + size
        metadata.path_index.extend([(start, end, i)] * size)
        start = end

    image_index = faiss.read_index(index)
    weight_name = 'checkpoint_0009'
    model, preprocessor = create_model(
        weight_name=weight_name,
        device='cuda', model_dir="/weights")
    model = model.to(device)
    model.eval()

    data_list = map(
        str, Path("/data/").resolve().glob("*.tar"))
    pipeline = [wds.SimpleShardList(data_list)]
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
        wds.batched(32, partial=True),
    ])
    dataset = wds.DataPipeline(*pipeline)
    ds = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=8,
    )
    features_cache = os.path.join(
        out, f"{dataset_name.replace('/','_')}_{index.replace('/','_')}_{weight_name.replace('/', '_')}.npy")
    meta_cache = os.path.join(
        out, f"{dataset_name.replace('/','_')}_{index.replace('/','_')}_{weight_name.replace('/', '_')}_meta.npy")
    print(features_cache)
    print(meta_cache)
    # create the folder if doesn't exists.
    Path(features_cache).resolve().parent.mkdir(parents=True, exist_ok=True)

    image_features_list = []
    meta_features = []
    # count = 0

    # Create a new queue
    queue = Queue()
    db_path = Path(out) / "test.db"
    if db_path.exists():
        # remove it using Pathlib
        db_path.unlink()
    db_name = str(db_path)

    # Create worker processes
    # Choose a number based on the number of cores in your machine
    num_workers = 4
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(queue,))
        processes.append(p)
        p.start()

    print("started finding similar images...")
    for X, meta in tqdm(ds, unit="batches"):
        file_meta = [{"tar_file": f"/data/{x['key'][:-4]}.tar",
                      "file_name": f"{x['key']}.jpg"} for x in meta]
        X = X.to(device)

        image_features = model(X)
        image_features = image_features.data.cpu()
        image_features = image_features.view(len(image_features), -1)
        image_features_list.append(image_features)
        meta_features.append(file_meta)

        image_features = np.concatenate(image_features_list)

        # print("Performing range search...")

        D, I_ = image_index.search(image_features, 1)
        # print("Score:", D.mean())
        L, D, I_ = image_index.range_search(
            image_features, cosine_similarity_threshold)

        # this should be submitted to the worker processes

        queue.put((L, D, I_, meta_features, metadata, db_name))

        image_features_list = []
        meta_features = []

    # Tell worker processes to stop when no more tasks are left
    for _ in range(num_workers):
        queue.put(None)

    # Wait for all of the workers to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    run(main)
