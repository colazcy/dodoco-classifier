import os
import json
import multiprocessing
import torch
import torch.nn as nn
import hparams
import dataset
import lightning as pl
from models.LangID import LangID
from torch.utils.data import DataLoader

hp = hparams.get_hparams()
num_frames = hp.clip_duration * hp.sample_rate
min_num_frames = hp.min_clip_duration * hp.sample_rate


class LitLangID(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LangID(num_lang=hp.num_lang, sample_rate=hp.sample_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        correct = (y_hat.argmax(dim=1) == y).type(torch.float).sum().item()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", correct / y.size(0), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hp.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=hp.lr_decay
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


torch.set_float32_matmul_precision(hp.float32_matmul_precision)

ds = dataset.load_dataset(
    path=hp.data_path,
    sample_rate=hp.sample_rate,
    test_size=hp.test_size,
    num_frames=num_frames,
    min_num_frames=min_num_frames,
)

cpu_count = multiprocessing.cpu_count()
train_loader = DataLoader(
    ds["train"], batch_size=hp.batch_size, num_workers=cpu_count, shuffle=True
)
test_loader = DataLoader(ds["test"], batch_size=hp.batch_size, num_workers=cpu_count)


model = LitLangID()

checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

trainer = pl.Trainer(
    max_epochs=hp.epochs,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    precision = hp.precision
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

best_model_path = checkpoint_callback.best_model_path

print(f"Best model saved at: {best_model_path}")

best_model = LitLangID.load_from_checkpoint(best_model_path).model
onnx_model = nn.Sequential(best_model,nn.Softmax(dim=1))
onnx_model.to(device="cpu")
onnx_model.eval()

dummy_input = torch.randn(1, num_frames)
onnx_program = torch.onnx.export(onnx_model,
    dummy_input,
    os.path.join(hp.save_path, "lang_id.onnx"),
    export_params=True,
    opset_version=20,
    verify=True
)

with open(os.path.join(hp.save_path, "lang_id.json"), "w") as f:
    json.dump(ds["lang_id"], f)
