# Add a model

IC-ThermBench is intended to make a new thermal predictor easy to compare with published baselines. A conventional PyTorch model requires one implementation file, one registry entry, and three thin scripts.

## 1. Implement the tensor contract

Create `model/MyNet.py`. The repository-level contract is:

```text
input   (B, X, Y, Z, P)
output  (B, X, Y, Z)
```

For the current release, `X = Y = 64`, `Z = 1`, and `P ∈ {3, 4, 7}`. Do not hard-code the input-channel count.

```python
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, in_channels, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x):
        # (B, X, Y, Z, P) -> (B, P, X, Y)
        b, nx, ny, nz, p = x.shape
        if nz != 1:
            raise ValueError("this 2D example expects Z=1")
        h = x.squeeze(3).permute(0, 3, 1, 2)
        # (B, 1, X, Y) -> (B, X, Y, Z=1)
        return self.net(h).permute(0, 2, 3, 1)
```

If the model already consumes channel-first tensors, adapting inside the model is usually the smallest and easiest-to-audit change.

## 2. Register the model and recipe

Add one entry to `MODEL_ZOO` in [`exp/exp_basic.py`](../exp/exp_basic.py):

```python
"MyNet": dict(
    prefix="mynet",
    epochs=100,
    batch_size=20,
    lr=1e-3,
    weight_decay=1e-4,
    sched=("step", 2, 0.9),
    finetune_lr=1e-4,
    build=lambda P, Z, G: (
        "model.MyNet:MyNet",
        dict(in_channels=P),
    ),
),
```

The registry is deliberately explicit:

- `prefix` names metric keys and must be unique.
- `epochs`, optimizer fields, and schedule define the reported training recipe.
- `finetune_lr` controls S5 target-domain adaptation.
- `build` receives the current input-channel count, layer count, and grid size.

Record intentional differences from existing recipes. Fairness is enforced at the data and evaluation interfaces, not by forcing every architecture to use an unsuitable optimizer.

## 3. Add command wrappers

Create `script/MyNet/train.sh`, `test.sh`, and `finetune.sh` from the U-FNO wrappers, then change only the model token:

```bash
mkdir -p script/MyNet
cp script/UFNO/train.sh script/MyNet/train.sh
cp script/UFNO/test.sh script/MyNet/test.sh
cp script/UFNO/finetune.sh script/MyNet/finetune.sh
sed -i.bak 's/^MODEL=.*/MODEL=MyNet/' script/MyNet/*.sh
rm script/MyNet/*.bak
chmod +x script/MyNet/*.sh
```

Then run:

```bash
bash script/MyNet/train.sh
bash script/MyNet/test.sh
bash script/MyNet/finetune.sh 10 50 100 250 500
```

S5 zero-shot evaluation and adaptation require an S4 checkpoint. Running the training wrapper without a scope argument creates S2, S3, and S4 checkpoints.

## Models with a different layout

The example converts layouts inside `forward`. If the shared experiment runner must perform the conversion, update these three functions together in [`exp/exp_operator.py`](../exp/exp_operator.py):

```text
_to_model
_from_model
_model_out_for_loss
```

Changing only inference conversion can make training loss and evaluation silently use different shapes.

## Models with a custom training stack

For a model that cannot use `exp/exp_operator.py`, set `builtin_loop=True` and implement:

```python
def train_model(
    self,
    x_train,
    y_train,
    epochs,
    batch_size,
    work_dir,
    epoch_log_fn=None,
    x_val=None,
    y_val=None,
    **kwargs,
):
    ...
    return x_normalizer, y_normalizer, folder
```

FNO is the reference implementation of this route. A completely separate framework can follow `exp/exp_thermfm.py`, provided it returns predictions in the common layout and uses the shared metrics.

## Join the aggregate scripts

After the single-model commands pass:

1. Add the model to `script/train_all.sh`.
2. Add it to `script/test_all.sh`.
3. Add it to `script/finetune_all.sh`.
4. Add its display order to `ORDER` in `utils/summarize.py`.

## Integration checklist

- [ ] Accepts P=3, P=4, and P=7 without source edits.
- [ ] Emits one temperature field per sample in kelvin after de-normalization.
- [ ] Uses the released split and does not fit normalizers on test or S5 samples.
- [ ] Uses `utils/metrics.py` for reported IC-ThermBench metrics.
- [ ] Loads an S4 checkpoint for S5 zero-shot evaluation.
- [ ] Reports architecture-specific optimizer and compute details.
- [ ] Writes a machine-readable result with `--output`.
- [ ] Documents any pretrained data or weights.
