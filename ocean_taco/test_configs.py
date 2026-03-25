"""Description of file."""

import os
from pathlib import Path

import cartopy
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def _configure_cartopy_dir(path: str):
    """Configure cartopy data directory."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    os.environ["CARTOPY_USER_DIR"] = str(p)
    cartopy.config["data_dir"] = str(p)


if __name__ == "__main__":
    _configure_cartopy_dir("./.cartopy")
    paths = [
        str(Path(__file__).parent / "configs" / "ssh_multi_time.yaml")
        # str(Path(__file__).parent / "configs" / "ssh_cond_single_time.yaml")
    ]
    for path in paths:
        print(f"Testing config: {path}")
        cfg = OmegaConf.load(path)

        # # Test Train
        # print("\n--- Testing Train ---")
        # ds_train = instantiate(cfg.train.dataset)
        # sampler_train = instantiate(cfg.train.sampler, dataset=ds_train)
        # dl_train = DataLoader(
        #     ds_train,
        #     batch_size=8,
        #     sampler=sampler_train,
        #     collate_fn=instantiate(cfg.train.collate_fn),
        #     num_workers=0,  # For debugging
        # )
        # for i, batch in enumerate(tqdm(dl_train, desc="Train Batches", total=len(sampler_train))):
        #     if i >= 10:  # Test first 5 batches
        #         break
        #     print(f"Train Batch {i}: Keys = {list(batch.keys())}, Shapes = {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}")

        # # Test Val
        # print("\n--- Testing Val ---")
        # ds_val = instantiate(cfg.val.dataset)
        # sampler_val = instantiate(cfg.val.sampler, dataset=ds_val)
        # dl_val = DataLoader(
        #     ds_val,
        #     batch_size=8,
        #     sampler=sampler_val,
        #     collate_fn=instantiate(cfg.val.collate_fn),
        #     num_workers=0,
        # )
        # for i, batch in enumerate(tqdm(dl_val, desc="Val Batches", total=len(sampler_val))):
        #     if i >= 10:
        #         break
        #     print(f"Val Batch {i}: Keys = {list(batch.keys())}, Shapes = {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}")

        # # Test Test (Two Samplers)
        print("\n--- Testing Test ---")
        ds_test = instantiate(cfg.test.dataset)

        # Gulf Stream
        sampler_test_gulf = instantiate(cfg.test.sampler_gulf_stream, dataset=ds_test)

        query = next(iter(sampler_test_gulf))

        sample = ds_test[query]

        ds_test.plot_sample(sample, save_path="test_regridding.png")

        dl_test_gulf = DataLoader(
            ds_test,
            batch_size=8,
            sampler=sampler_test_gulf,
            collate_fn=instantiate(cfg.test.collate_fn),
            num_workers=0,
        )
        print("Test Gulf Stream:")
        for i, batch in enumerate(
            tqdm(dl_test_gulf, desc="Test Gulf Batches", total=len(sampler_test_gulf))
        ):
            if i >= 10:
                break
            print(
                f"Test Gulf Batch {i}: Keys = {list(batch.keys())}, Shapes = {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}"
            )

        # Kuroshio
        sampler_test_kuro = instantiate(cfg.test.sampler_kuroshio, dataset=ds_test)
        dl_test_kuro = DataLoader(
            ds_test,
            batch_size=8,
            sampler=sampler_test_kuro,
            collate_fn=instantiate(cfg.test.collate_fn),
            num_workers=4,
        )
        print("Test Kuroshio:")
        for i, batch in enumerate(
            tqdm(dl_test_kuro, desc="Test Kuro Batches", total=len(sampler_test_kuro))
        ):
            # if i >= 10:
            #     break
            print(
                f"Test Kuro Batch {i}: Keys = {list(batch.keys())}, Shapes = {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}"
            )

        print(f"Config {path} tested successfully!")
