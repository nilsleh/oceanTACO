"""Description of file."""

from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

cfg = OmegaConf.load(
    "nils/oceanTACO-experiments/configs/oceanTACO/overfit_sst.yaml"
)
datamodule = instantiate(cfg.datamodule, num_workers=4)
datamodule.setup()
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

for batch in tqdm(train_loader):
    continue

for batch in tqdm(val_loader):
    continue

for batch in tqdm(test_loader):
    continue


input_vars = datamodule.input_variables
target_vars = datamodule.target_variables

for var in input_vars:
    print(batch["inputs"][var].shape)

for var in target_vars:
    print(batch["targets"][var].shape)

print(f"Batch keys: {batch.keys()}")


print(0)
