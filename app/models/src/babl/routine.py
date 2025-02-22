import kubeflow.katib as katib
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import logging
logger = logging.getLogger(__name__)


class Routine(pl.LightningModule):
    def __init__(self, model, vocab_size=32128, hpo=True):
        super().__init__()
        self.model = model
        self.lr = 1e-3
        self.vocab_size= vocab_size
        self.hpo=hpo
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        decoder_attention_mask, 
    ):
        y_hat = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, labels=labels)
        
        # print(f"forward(): {y_hat=}")
        return y_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # print(f"keys = {batch.keys()}")
        # print(f"{batch=}")
        y = batch['labels']
        y_hat = self(**batch)
        y_onehot = F.one_hot(y, num_classes=self.vocab_size)
        y = y_onehot.float()
        losses = []
        # computing cross-entropy on per-token basis and averaging the loss. 
        for tok in range(y_hat.logits.shape[1]):
            # print("Per-token loss cross entropy")
            loss = F.cross_entropy(y_hat.logits[:,tok,:] , y[:,tok,:])
            # print(loss)
            # loss = F.nll_loss(y_hat[:,tok,:] , y[:,tok,:])
            losses.append(loss)

        loss  = torch.tensor(losses,  requires_grad=True).mean()
        # dummy metrics


        # calculating exact matches 
        y_hat = F.softmax(y_hat.logits, dim=-1)
        a = y_hat.argmax(1)
        a.to(y_hat.device)
        y_hat = torch.zeros(y_hat.shape,  device=y_hat.device).scatter(1, a.unsqueeze(1), 1.0)
        # y_onehot = F.one_hot(y, num_classes=VOCAB_SIZE)
        # y = y_onehot.float()
        matches = (y == y_hat).int()
        correct = matches.sum()
        tot = torch.prod(torch.tensor(matches.shape))
        metrics_dict = {"loss": loss, "train_EM": (correct/tot).item(), "train_F1": 0.9}
        if self.hpo:
            metrics_dict_katib ={"train_loss": loss, "train_EM": (correct/tot).item(), "train_F1": 0.9}
            katib.report_metrics(metrics_dict_katib)
        
        # print(metrics_dict)
        self.training_step_outputs.append(metrics_dict)
        return metrics_dict



    def on_train_epoch_end(self):
        results = {
            "loss": torch.tensor(
                [x["loss"] for x in self.training_step_outputs]
            ).mean(),
            "F1": torch.tensor(
                [x["train_F1"] for x in self.training_step_outputs]
            ).mean(),
            "EM": torch.tensor(
                [x["train_EM"] for x in self.training_step_outputs]
            ).mean(),
        }
        # self.log(f"LR",self.lr, on_epoch=True, prog_bar=True, logger=True)
        for k, v in results.items():
            self.log(
                f"train_{k}",
                v,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        y = batch['labels']
        y_hat = self(**batch)
        y_onehot = F.one_hot(y, num_classes=self.vocab_size)
        y = y_onehot.float()
        losses = []
        # computing cross-entropy on per-token basis and averaging the loss. 
        for tok in range(y_hat.logits.shape[1]):
            loss = F.cross_entropy(y_hat.logits[:,tok,:] , y[:,tok,:])
            # print(loss)
            losses.append(loss)
        loss  = torch.tensor(losses).mean()



        # calculating exact matches 
        y_hat = F.softmax(y_hat.logits, dim=-1)
        a = y_hat.argmax(1)
        y_hat = torch.zeros(y_hat.shape,  device=y_hat.device).scatter(1, a.unsqueeze(1), 1.0)
        matches = (y == y_hat).int()
        correct = matches.sum()
        tot = torch.prod(torch.tensor(matches.shape))
        


        # dummy metrics
        metrics_dict = {"val_loss": loss.item(), "val_EM": (correct/tot).item(), "val_F1": 0.9}
        if self.hpo:
            katib.report_metrics(metrics_dict)
        self.validation_step_outputs.append(metrics_dict)
        return metrics_dict



    def on_validation_epoch_end(self):
        results = {
            "loss": torch.tensor(
                [x["val_loss"] for x in self.validation_step_outputs]
            ).mean(),
            "EM": torch.tensor(
                [x["val_EM"] for x in self.validation_step_outputs]
            ).mean(),
            "F1": torch.tensor(
                [x["val_F1"] for x in self.validation_step_outputs]
            ).mean(),
        }
        for k, v in results.items():
            self.log(
                f"val_{k}", v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
            )
            # self.log(f"val_{k}", v, on_epoch=True, prog_bar=True) # , logger=True)

    def test_step(self, batch, batch_idx):
        # x = batch["x"]
        
        
        y = batch["labels"]
        y_hat = self(**batch)
        
        # calculating exact matches 
        y_hat = F.softmax(y_hat.logits, dim=-1)
        a = y_hat.argmax(1)
        a.to(y_hat.device)
        y_hat = torch.zeros(y_hat.shape,  device=y_hat.device).scatter(1, a.unsqueeze(1), 1.0)
        y_onehot = F.one_hot(y, num_classes=self.vocab_size)
        y = y_onehot.float()
        matches = (y == y_hat).int()
        correct = matches.sum()
        tot = torch.prod(torch.tensor(matches.shape))
        
        
        metrics_dict = {
            "test_EM": (correct/tot).item(),
            "test_F1": 0.8,
        }
        self.test_step_outputs.append(metrics_dict)
        return metrics_dict

    def on_test_epoch_end(self):
        results = {
            "F1": torch.tensor([x["test_F1"] for x in self.test_step_outputs]).mean(),
            "EM": torch.tensor([x["test_EM"] for x in self.test_step_outputs]).mean(),
        }

        for k, v in results.items():
            self.log(
                f"test_{k}",
                v,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def configure_optimizers(self):

        # special scheduler for transformers
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=0.001,  # self.cfg_fitting.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.05,
        )
        return {
            "optimizer": optimizer,
            "monitor": "val_loss",
        }
