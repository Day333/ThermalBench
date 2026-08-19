"""Loading checkpoints produced by the legacy repositories.

Historical checkpoints were written as `torch.save([x_normalizer, model, y_normalizer])`,
i.e. whole objects, so the pickle records each class by its **original module path**:

    ufno.Net3d              sau_fno.SAUNet3d        unet.UNet
    deeponet.DeepONetSup    models.fourier_3d.FNO3d models.normalize.normalize
    __main__._PerChannelNorm   <- the per-channel normalizer defined inline in the
                                  training scripts

This project moved those classes under model/ and layers/, so the module paths changed
and a plain torch.load raises ModuleNotFoundError. `install()` registers the old paths
as aliases of the new modules so pickle can find them.

The model classes keep the historical attribute names and forward numerics, so an
object deserialized this way evaluates identically to the one trained back then --
which is precisely what makes "new code reproduces the old numbers" a meaningful
check, and the regression suite verifies it against the released checkpoints.

Checkpoints written by new training runs no longer store whole objects; they store a
state_dict (see utils/tools.save_checkpoint) and depend on no class path at all.
"""
import sys
import types


class _PerChannelNorm:
    """Placeholder used only to unpickle the legacy __main__._PerChannelNorm.

    The old training scripts (ufno_train / unet_train / sau_fno_train /
    deeponet_train) each defined a class of this name in __main__. The
    implementations agree with each other and with
    layers.normalize.per_channel_normalize (statistics over dim=(0,1,2,3), std==0
    replaced by 1). Deserialization only needs the class to exist; forward/inverse
    below match the original numerics.
    """

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    __call__ = forward


def install():
    """Register the legacy module aliases. Safe to call repeatedly."""
    import layers.normalize as _norm
    import model.DeepONet as _don
    import model.FNO as _fno
    import model.SAUFNO as _sau
    import model.UFNO as _ufno
    import model.UNet as _unet

    # top-level module names from the old repos -> the new modules
    for old, new in [("ufno", _ufno), ("sau_fno", _sau), ("unet", _unet),
                     ("deeponet", _don)]:
        sys.modules.setdefault(old, new)

    # the fno repo was a package, models.xxx
    if "models" not in sys.modules:
        pkg = types.ModuleType("models")
        pkg.__path__ = []
        sys.modules["models"] = pkg
    sys.modules.setdefault("models.normalize", _norm)

    # in models.fourier_3d the old name was SpectralConv3d, the new one FNOSpectralConv3d
    if "models.fourier_3d" not in sys.modules:
        shim = types.ModuleType("models.fourier_3d")
        shim.FNO3d = _fno.FNO3d
        shim.SpectralConv3d = _fno.FNOSpectralConv3d
        sys.modules["models.fourier_3d"] = shim

    # the normalizer class the training scripts defined inline
    main = sys.modules["__main__"]
    if not hasattr(main, "_PerChannelNorm"):
        # give it real nn.Module behaviour (the parameters are already in the
        # pickle's __dict__)
        main._PerChannelNorm = type(
            "_PerChannelNorm", (_norm.per_channel_normalize,), {"__init__": lambda self: None})


def patch_legacy_attrs(model):
    """Fill in attributes missing from older whole-object checkpoints.

    Currently a no-op: the model classes only read attributes that every historical
    checkpoint already carries. Kept as the extension point for future pickled-object
    incompatibilities.
    """
    return model
