from typing import Any

import einops
import torch
import torchvision
from lightning import LightningModule
from PIL import Image
from torchmetrics import MeanMetric


class GenerativeModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.net(*args, **kwargs)

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        return self.net.sample(num_samples, **kwargs)

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.net.compute_loss(batch, batch_idx)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch, batch_idx)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch, batch_idx)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self._log_images("val")
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch, batch_idx)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self._log_images("test")
        return loss

    def _log_images(self, stage: str) -> None:
        images = self.sample(8)
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=True, pad_value=1)
        grid_np = einops.rearrange(grid, "c h w -> h w c").cpu().numpy()
        grid_np = (grid_np * 255).astype("uint8")
        grid_pil = Image.fromarray(grid_np)

        for logger in self.loggers:
            if hasattr(logger, "experiment"):
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    grid_pil.save(tmp_path)

                    artifact_path = f"{stage}/samples_epoch_{self.current_epoch}.png"
                    logger.experiment.log_artifact(
                        run_id=logger.run_id,
                        local_path=tmp_path,
                        artifact_path=artifact_path,
                    )

                    os.unlink(tmp_path)

    def configure_optimizers(self) -> dict[str, Any] | list[dict[str, Any]]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
