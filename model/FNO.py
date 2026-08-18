"""FNO3d (the FNO baseline).

NOTE: unlike every other model, this class **carries its own train_model / test_model**
instead of having them lifted into exp/. That is deliberate: FNO's training loop uses
the repo's bundled models/Adam.py (not torch.optim.Adam) plus its own validation and
early-stopping logic, and rewriting it as a generic loop would necessarily change the
numbers. The benchmark is judged on reproducing existing results, so fidelity wins over
uniformity -- exp/exp_operator.py calls this method through a dedicated branch.

Dropout defaults to 0.1; the benchmark sets FNO_DROPOUT=0 to match U-FNO, and evaluation
takes a separate deterministic path (see exp/) rather than the built-in MC-dropout
predict(). Class body copied verbatim from fno/models/fourier_3d.py.
"""
import os
from timeit import default_timer

import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from layers.fourier import FNOSpectralConv3d
from layers.optim import Adam
from layers.normalize import cal_rmse, make_norm, normalize
from utils.losses import LpLoss, thermal_combined_loss


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=4, dr=None):
        super().__init__()
        # defaults to 0.1 (original behaviour); FNO_DROPOUT=0 disables it to match U-FNO
        if dr is None:
            dr = float(os.environ.get("FNO_DROPOUT", 0.1))
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels

        self.fc0 = nn.Linear(in_channels, self.width)
        self.conv0 = FNOSpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.conv1 = FNOSpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.conv2 = FNOSpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.conv3 = FNOSpectralConv3d(self.width, self.width, modes1, modes2, modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.drop1 = nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)
        self.drop3 = nn.Dropout(dr)
        self.drop4 = nn.Dropout(dr)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = self.drop1(x)
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = self.drop2(x)
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.drop3(x)
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        return x

    def _evaluate_loss(self, data_loader, loss_type="base", thermal_lam=None):
        self.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.cuda(), y.cuda()
                out = self(x.to(dtype=torch.double))
                out = out.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                loss = thermal_combined_loss(out, y, x, loss_type, thermal_lam)
                val_loss_sum += loss.item() * x.shape[0]
                val_sample_count += x.shape[0]
        self.train()
        return val_loss_sum / max(val_sample_count, 1)

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
        loss_type="base",
        thermal_lam=None,
    ):
        optimizer = Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        # keeps the original behaviour by default; the env vars align the schedule
        # with U-FNO's (FNO_LR_STEP=2 FNO_LR_GAMMA=0.9)
        _step = int(os.environ.get("FNO_LR_STEP", 100))
        _gamma = float(os.environ.get("FNO_LR_GAMMA", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=_step, gamma=_gamma)

        x_normalizer = make_norm(x_train)
        y_normalizer = make_norm(y_train)
        x_train = x_normalizer.forward(x_train)
        y_train = y_normalizer.forward(y_train)
        has_val = x_val is not None and y_val is not None
        if has_val:
            x_val = x_normalizer.forward(x_val)
            y_val = y_normalizer.forward(y_val)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = None
        if has_val:
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_val, y_val),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
        myloss = LpLoss(size_average=False)
        y_normalizer.cuda()

        t1 = default_timer()
        self.train()
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                out = self(x.to(dtype=torch.double))
                out = out.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                # thermal combined loss in NORMALIZED space (no inverse); y is
                # normalized here (train_loader feeds y_normalizer.forward(y))
                loss = thermal_combined_loss(out, y, x, loss_type, thermal_lam)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += loss.item() * x.shape[0]
                epoch_sample_count += x.shape[0]
            scheduler.step()
            epoch_train_loss = epoch_loss_sum / max(epoch_sample_count, 1)
            log_payload = {
                "train/loss": epoch_train_loss,
                "train/epoch": epoch + 1,
            }
            if val_loader is not None:
                epoch_val_loss = self._evaluate_loss(val_loader, loss_type, thermal_lam)
                log_payload["val/loss"] = epoch_val_loss
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"train/loss: {epoch_train_loss:.6f}, val/loss: {epoch_val_loss:.6f}"
                )
            else:
                print(f"Epoch [{epoch + 1}/{epochs}] train/loss: {epoch_train_loss:.6f}")
            if epoch_log_fn is not None:
                epoch_log_fn(log_payload)

        print("training times:", default_timer() - t1)
        folder_path = work_dir if os.path.isabs(work_dir) else os.path.join(os.getcwd(), work_dir)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return x_normalizer, y_normalizer, folder_path

    def test_model(self, x_test, y_test, x_normalizer, y_normalizer):
        x_test = x_normalizer.forward(x_test)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test),
            batch_size=20,
            shuffle=False,
            drop_last=False,
        )
        rmse_total = 0
        r2_total = 0
        maxx_total = 0
        differ_total = 0
        mape_total = 0
        pape_total = 0
        maxterr_total = 0   # max_temperature_error: |max(pred)-max(true)| (hotspot peak, K)
        topk_total = 0      # topk_temperature_difference over hottest TRUE pixels (K)
        topk = 50
        num = 0

        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            mean_out, _ = self.predict(x, y_normalizer)
            out = torch.cat(mean_out, dim=0).cpu().detach()
            y_np = y.cpu().detach().numpy()
            for i in range(x.shape[0]):
                rmse = 0
                maxx = 0
                differ = 0
                mape = 0
                pape = 0
                maxterr = 0
                topk_diff = 0
                for l in range(y_np.shape[-1]):
                    rmse += cal_rmse(out[i, ..., l], y_np[i, ..., l])
                    maxx += torch.max(torch.abs(out[i, ..., l] - y_np[i, ..., l]))
                    differ += torch.mean(torch.abs(out[i, ..., l] - y_np[i, ..., l]))
                    mape += (
                        torch.mean(torch.abs(out[i, ..., l] - y_np[i, ..., l]) / y_np[i, ..., l]) * 100
                    )
                    pape += (
                        torch.max(torch.abs(out[i, ..., l] - y_np[i, ..., l]) / y_np[i, ..., l]) * 100
                    )
                    # hotspot metrics (denormalized K; mirrors Therm-FM evaluate.py)
                    pred_l = out[i, ..., l]
                    true_l = torch.from_numpy(y_np[i, ..., l])
                    maxterr += float(torch.abs(pred_l.max() - true_l.max()))
                    kk = min(topk, true_l.numel())
                    tidx = torch.topk(true_l.flatten(), kk).indices
                    topk_diff += float(torch.mean(torch.abs(pred_l.flatten()[tidx] - true_l.flatten()[tidx])))
                depth = y_np.shape[-1]
                rmse_total += rmse / depth
                maxx_total += maxx / depth
                differ_total += differ / depth
                mape_total += mape / depth
                pape_total += pape / depth
                maxterr_total += maxterr / depth
                topk_total += topk_diff / depth
                r2 = 0
                for l in range(depth):
                    r2 += r2_score(y_np[i, ..., l].flatten(), out[i, ..., l].flatten()) # asymmetric metric: the argument order (y_true, y_pred) must not be swapped
                r2_total += r2 / depth
                num += 1

        metrics = {
            "rmse": float(rmse_total / num),
            "r2": float(r2_total / num),
            "max.terr": float(maxx_total / num),
            "avg.terr": float(differ_total / num),
            "mape": float(mape_total / num),
            "pape": float(pape_total / num),
            "max_temperature_error": float(maxterr_total / num),
            "topk50_temperature_difference": float(topk_total / num),
        }
        print("rmse:", metrics["rmse"])
        print("r2:", metrics["r2"])
        print("max.terr:", metrics["max.terr"])
        print("avg.terr:", metrics["avg.terr"])
        print("mape:", metrics["mape"])
        print("pape:", metrics["pape"])
        print("max_temperature_error:", metrics["max_temperature_error"])
        print("topk50_temperature_difference:", metrics["topk50_temperature_difference"])
        return metrics

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def predict(self, x, y_normalizer, samples=20):
        pred_mean = []
        pred_std = []
        self.enable_dropout()
        for i in range(x.shape[0]):
            test_x = x[i, ...].unsqueeze(0)
            result = []
            for _ in range(samples):
                out = self(test_x).reshape(
                    test_x.shape[0], test_x.shape[1], test_x.shape[2], test_x.shape[3]
                )
                out = y_normalizer.inverse(out).cpu().detach()
                result.append(out)
            stacked_tensor = torch.stack(result, axis=0)
            pred_mean.append(torch.mean(stacked_tensor, axis=0))
            pred_std.append(torch.std(stacked_tensor, axis=0))
        return pred_mean, pred_std
