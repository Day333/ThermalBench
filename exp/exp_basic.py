"""Model registry -- each model's **training recipe** lives here. Change one and the
published numbers stop reproducing.

The table below was reconciled entry by entry against each model's original training
script. The differences are real; do not "tidy them up" into a uniform config:

  UNet   uses Adam with **no weight_decay** and **no lr schedule**, and runs **200**
         epochs; every other model uses wd=1e-4 + StepLR(step=2, gamma=0.9) + 100 epochs.
  U-FNO / SAU-FNO pass `foreach=False` to Adam explicitly, DeepONet does not.
         `foreach` changes how multi-tensor updates are fused, which shows up in the
         last digits.
  FNO    does not use this file's recipe at all: it uses the vendored models/Adam.py and
         keeps its training loop in model/FNO.py's train_model. See exp/exp_operator.py.

`ckpt_tag` is the directory-name fragment of the historical checkpoints, used to locate
old weights for reproduction checks.
`prefix` is the key prefix of the metrics dict; it must match the original or RMSE and
friends cannot be read back out.
`finetune_lr` is the default learning rate for level5 few-shot fine-tuning -- **one tenth
of each model's own training lr**, not one shared value. Therm-FM trains at only 5e-5;
giving it the FNO family's 1e-4 diverges immediately (measured RMSE 16.01 -> 31.34 after
a single epoch).
"""
MODEL_ZOO = {
    "FNO": dict(
        prefix="fno_det", ckpt_tag="fno", epochs=100, batch_size=20, finetune_lr=1e-4,
        builtin_loop=True,
        # FNO3d reads these three from the environment. The defaults (dropout=0.1,
        # StepLR(100, 0.5)) are NOT the benchmark configuration: the benchmark turns
        # dropout off to match U-FNO and switches to U-FNO's StepLR(2, 0.9). An FNO
        # trained without these three will not reproduce the published numbers.
        # FNO_DROPOUT is read inside FNO3d.__init__, so it must be set before the
        # model is constructed.
        env={"FNO_DROPOUT": "0", "FNO_LR_STEP": "2", "FNO_LR_GAMMA": "0.9"},
        build=lambda P, Z, G: ("model.FNO:FNO3d", dict(
            modes1=12, modes2=12, modes3=min(2, Z // 2 + 1), width=72, in_channels=P)),
    ),
    "UFNO": dict(
        prefix="ufno", ckpt_tag="ufno", epochs=100, batch_size=20, finetune_lr=1e-4,
        lr=1e-3, weight_decay=1e-4, foreach=False, sched=("step", 2, 0.9),
        build=lambda P, Z, G: ("model.UFNO:Net3d", dict(
            modes1=10, modes2=10, modes3=min(10, Z // 2 + 1), width=36, in_channels=P)),
    ),
    "SAUFNO": dict(
        prefix="sau", ckpt_tag="sau_fno", epochs=100, batch_size=20, finetune_lr=1e-4,
        lr=1e-3, weight_decay=1e-4, foreach=False, sched=("step", 2, 0.9),
        build=lambda P, Z, G: ("model.SAUFNO:SAUNet3d", dict(
            modes1=10, modes2=10, modes3=min(10, Z // 2 + 1), width=36, in_channels=P)),
    ),
    "UNet": dict(
        prefix="unet", ckpt_tag="unet", epochs=200, batch_size=20, finetune_lr=1e-4,
        lr=1e-3, weight_decay=0.0, sched=None,
        build=lambda P, Z, G: ("model.UNet:UNet", dict(in_channels=P, out_channels=1)),
    ),
    "DeepONet": dict(
        prefix="deeponet", ckpt_tag="deeponet", epochs=100, batch_size=20, finetune_lr=1e-4,
        lr=1e-3, weight_decay=1e-4, sched=("step", 2, 0.9),
        build=lambda P, Z, G: ("model.DeepONet:DeepONetSup", dict(in_channels=P, grid=G)),
    ),
    # The three Therm-FM sizes run a completely different pipeline (HuggingFace Trainer
    # + accelerate, multi-GPU). Their config lives in model/thermfm_configs/*.yaml;
    # see exp/exp_thermfm.py.
    "ThermFM-T": dict(prefix="thermfm", ckpt_tag="T", scot=True, size="T",
                     finetune_lr=5e-6),
    "ThermFM-B": dict(prefix="thermfm", ckpt_tag="B", scot=True, size="B",
                     finetune_lr=5e-6),
    "ThermFM-L": dict(prefix="thermfm", ckpt_tag="L", scot=True, size="L",
                     finetune_lr=5e-6),
}


# "Which datasets do not take part in training" is a property of the data, so it is
# defined in data_provider; re-exported here for convenience.
from data_provider.data_factory import PURE_EVAL, source_level  # noqa: E402,F401


OPERATOR_MODELS = [k for k, v in MODEL_ZOO.items() if not v.get("scot")]
SCOT_MODELS = [k for k, v in MODEL_ZOO.items() if v.get("scot")]


def build_model(name, P, Z, G):
    """Instantiate a model from the registry. P=input channels, Z=layers, G=grid side.

    `build` returns ("module.path:ClassName", ctor_kwargs). Using a string instead of a
    direct import is what keeps "add a new model" down to editing MODEL_ZOO alone --
    there is no second place that needs an import branch.
    """
    import importlib

    cls_path, kwargs = MODEL_ZOO[name]["build"](P, Z, G)
    module, _, cls_name = cls_path.partition(":")
    return getattr(importlib.import_module(module), cls_name)(**kwargs)
