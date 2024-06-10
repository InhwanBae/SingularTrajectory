import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from . import *


class STTrainer:
    r"""Base class for all Trainers"""

    def __init__(self, args, hyper_params):
        print("Trainer initiating...")

        # Reproducibility
        reproducibility_settings(seed=0)

        self.args, self.hyper_params = args, hyper_params
        self.model, self.optimizer, self.scheduler = None, None, None
        self.loader_train, self.loader_val, self.loader_test = None, None, None
        self.dataset_dir = hyper_params.dataset_dir + hyper_params.dataset + '/'
        self.checkpoint_dir = hyper_params.checkpoint_dir + '/' + args.tag + '/' + hyper_params.dataset + '/'
        print("Checkpoint dir:", self.checkpoint_dir)
        self.log = {'train_loss': [], 'val_loss': []}
        self.stats_func, self.stats_meter = None, None
        self.reset_metric()

        if not args.test:
            # Save arguments and configs
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            with open(self.checkpoint_dir + 'args.pkl', 'wb') as fp:
                pickle.dump(args, fp)

            with open(self.checkpoint_dir + 'config.pkl', 'wb') as fp:
                pickle.dump(hyper_params, fp)

    def init_descriptor(self):
        # Singular space initialization
        print("Singular space initialization...")
        obs_traj, pred_traj = self.loader_train.dataset.obs_traj, self.loader_train.dataset.pred_traj
        obs_traj, pred_traj = augment_trajectory(obs_traj, pred_traj)
        self.model.calculate_parameters(obs_traj, pred_traj)
        print("Anchor generation...")
    
    def init_adaptive_anchor(self, dataset):
        print("Adaptive anchor initialization...")
        dataset.anchor = self.model.calculate_adaptive_anchor(dataset)

    def train(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def valid(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

    def fit(self):
        print("Training started...")
        for epoch in range(self.hyper_params.num_epochs):
            self.train(epoch)
            self.valid(epoch)

            if self.hyper_params.lr_schd:
                self.scheduler.step()

            # Save the best model
            if epoch == 0 or self.log['val_loss'][-1] < min(self.log['val_loss'][:-1]):
                self.save_model()

            print(" ")
            print("Dataset: {0}, Epoch: {1}".format(self.hyper_params.dataset, epoch))
            print("Train_loss: {0:.8f}, Val_los: {1:.8f}".format(self.log['train_loss'][-1], self.log['val_loss'][-1]))
            print("Min_val_epoch: {0}, Min_val_loss: {1:.8f}".format(np.array(self.log['val_loss']).argmin(),
                                                                     np.array(self.log['val_loss']).min()))
            print(" ")
        print("Done.")

    def reset_metric(self):
        self.stats_func = {'ADE': compute_batch_ade, 'FDE': compute_batch_fde}
        self.stats_meter = {x: AverageMeter() for x in self.stats_func.keys()}

    def get_metric(self):
        return self.stats_meter

    def load_model(self, filename='model_best.pth'):
        model_path = self.checkpoint_dir + filename
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, filename='model_best.pth'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        model_path = self.checkpoint_dir + filename
        torch.save(self.model.state_dict(), model_path)


class STSequencedMiniBatchTrainer(STTrainer):
    r"""Base class using sequenced mini-batch training strategy"""

    def __init__(self, args, hyper_params):
        super().__init__(args, hyper_params)

        # Dataset preprocessing
        obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
        self.loader_train = get_dataloader(self.dataset_dir, 'train', obs_len, pred_len, batch_size=1, skip=skip)
        self.loader_val = get_dataloader(self.dataset_dir, 'val', obs_len, pred_len, batch_size=1)
        self.loader_test = get_dataloader(self.dataset_dir, 'test', obs_len, pred_len, batch_size=1)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0
        is_first_loss = True

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            self.optimizer.zero_grad()

            output = self.model(obs_traj, pred_traj)

            loss = output["loss_euclidean_ade"]
            loss[torch.isnan(loss)] = 0

            if (cnt + 1) % self.hyper_params.batch_size != 0 and (cnt + 1) != len(self.loader_train):
                if is_first_loss:
                    is_first_loss = False
                    loss_cum = loss
                else:
                    loss_cum += loss

            else:
                is_first_loss = True
                loss_cum += loss
                loss_cum /= self.hyper_params.batch_size
                loss_cum.backward()

                if self.hyper_params.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)

                self.optimizer.step()
                loss_batch += loss_cum.item()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj, pred_traj)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class STCollatedMiniBatchTrainer(STTrainer):
    r"""Base class using collated mini-batch training strategy"""

    def __init__(self, args, hyper_params):
        super().__init__(args, hyper_params)

        # Dataset preprocessing
        batch_size = hyper_params.batch_size
        obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
        self.loader_train = get_dataloader(self.dataset_dir, 'train', obs_len, pred_len, batch_size=batch_size, skip=skip)
        self.loader_val = get_dataloader(self.dataset_dir, 'val', obs_len, pred_len, batch_size=batch_size)
        self.loader_test = get_dataloader(self.dataset_dir, 'test', obs_len, pred_len, batch_size=1)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            self.optimizer.zero_grad()

            output = self.model(obs_traj, pred_traj)

            loss = output["loss_euclidean_ade"]
            loss[torch.isnan(loss)] = 0
            loss_batch += loss.item()

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj, pred_traj)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class STTransformerDiffusionTrainer(STCollatedMiniBatchTrainer):
    r"""SingularTrajectory model trainer"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)
        cfg = DotDict({'scheduler': 'ddim', 'steps': 10, 'beta_start': 1.e-4, 'beta_end': 5.e-2, 'beta_schedule': 'linear', 
                       'k': hyper_params.k, 's': hyper_params.num_samples})
        predictor_model = base_model(cfg).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=hyper_params.lr,
                                           weight_decay=hyper_params.weight_decay)

        if hyper_params.lr_schd:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                             step_size=hyper_params.lr_schd_step,
                                                             gamma=hyper_params.lr_schd_gamma)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0

        if self.loader_train.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_train.dataset)

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = batch["obs_traj"].cuda(non_blocking=True), batch["pred_traj"].cuda(non_blocking=True)
            adaptive_anchor = batch["anchor"].cuda(non_blocking=True)
            scene_mask, seq_start_end = batch["scene_mask"].cuda(non_blocking=True), batch["seq_start_end"].cuda(non_blocking=True)
            
            self.optimizer.zero_grad()

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info=additional_information)

            loss = output["loss_euclidean_ade"]
            loss[torch.isnan(loss)] = 0
            loss_batch += loss.item()

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        if self.loader_val.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_val.dataset)
        
        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = batch["obs_traj"].cuda(non_blocking=True), batch["pred_traj"].cuda(non_blocking=True)
            adaptive_anchor = batch["anchor"].cuda(non_blocking=True)
            scene_mask, seq_start_end = batch["scene_mask"].cuda(non_blocking=True), batch["seq_start_end"].cuda(non_blocking=True)

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info=additional_information)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        if self.loader_test.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_test.dataset)

        for cnt, batch in enumerate(tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene")):
            obs_traj, pred_traj = batch["obs_traj"].cuda(non_blocking=True), batch["pred_traj"].cuda(non_blocking=True)
            adaptive_anchor = batch["anchor"].cuda(non_blocking=True)
            scene_mask, seq_start_end = batch["scene_mask"].cuda(non_blocking=True), batch["seq_start_end"].cuda(non_blocking=True)

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, adaptive_anchor, addl_info=additional_information)

            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}
