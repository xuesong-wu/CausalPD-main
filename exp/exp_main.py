from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import CausalPD
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, RMSE
from layers.intervene import separate_patches
from utils.masking import compute_attention_masks
# from neuralforecast.losses.numpy import rmse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
# import wandb
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'CausalPD': CausalPD,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(f"=== Model: {self.args.model} ===")
        # total_params = 0
        # for name, param in model.named_parameters():
        #     num = param.numel()
        #     total_params += num
        #     shape_str = str(tuple(param.size()))
        #     print(f"{name:50}  shape={shape_str:15}  params={num}")
        # print(f"{'Total parameters':50}  {total_params}\n")
        # print(f"Approx. model size (float32): {total_params * 4 / (1024 ** 2):.2f} MB")

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mae':
            criterion = nn.L1Loss()
        elif self.args.loss == 'mse':
            criterion = nn.MSELoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        preds = []
        trues = []
        scaler = vali_data.scaler
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                if self.args.meta_dim > 0:  # With meta data
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data = batch_data
                else:  # Without meta data
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    meta_data = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if meta_data is not None:
                    meta_data = meta_data.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'PD' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, meta_data=meta_data)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'PD' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, meta_data=meta_data)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                original_shape = batch_y.shape
                pred_np = scaler.inverse_transform(outputs.reshape(-1, original_shape[-1]))
                true_np = scaler.inverse_transform(batch_y.reshape(-1, original_shape[-1]))

                pred1 = pred_np.reshape(original_shape)
                true1 = true_np.reshape(original_shape)
                preds.append(pred1)
                trues.append(true1)
        preds = np.array(preds).reshape(-1, pred1.shape[-2], pred1.shape[-1])
        trues = np.array(trues).reshape(-1, true1.shape[-2], true1.shape[-1])
        rmse_val = RMSE(preds, trues)
        self.model.train()
        return rmse_val

    # ============================= Stage 1: Conventional Training =============================
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                          steps_per_epoch=train_steps,
                                          pct_start=self.args.pct_start,
                                          epochs=self.args.train_epochs,
                                          max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.meta_dim > 0:  # With meta data
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data = batch_data
                else:  # Without meta data
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    meta_data = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if meta_data is not None:
                    meta_data = meta_data.float().to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'PD' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, meta_data=meta_data)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'PD' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, meta_data=meta_data)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Learning rate scheduling
            if self.args.lradj != 'PD':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                current_lr = model_optim.param_groups[0]['lr']
                decay_factor = self.args.intervention_lr_decay
                new_lr = current_lr * decay_factor
                for param_group in model_optim.param_groups:
                    param_group['lr'] = new_lr
                print('Updating learning rate to {:.6f}'.format(new_lr))


        # ============================= Stage 2: Intervention Training if enabled ============================= 
        if self.args.use_intervention:
            print("=== Stage 2: Intervention Training ===")
            # Load best model from conventional training
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

            # Compute causal and non-causal masks for intervention
            self.causal_mask, self.noncausal_mask = compute_attention_masks(
                self.model, setting, self.args.attention_threshold)
            print("Causal and non-causal masks computed.")

            # Setup optimizer and scheduler for intervention training
            current_lr = model_optim.param_groups[0]['lr']
            intervention_lr = current_lr * self.args.intervention_lr_scale
            print(f"Adjusting learning rate from {current_lr:.6f} to {intervention_lr:.6f}")
            
            for param_group in model_optim.param_groups:
                param_group['lr'] = intervention_lr
            

            if hasattr(scheduler, 'max_lr'):
                scheduler.max_lr = intervention_lr
                scheduler.epochs = 15
            elif hasattr(scheduler, 'base_lrs'):
                for i in range(len(scheduler.base_lrs)):
                    scheduler.base_lrs[i] = intervention_lr

            criterion = self._select_criterion()
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()
                
            # Intervention training loop
            for epoch in range(self.args.intervention_epochs):
                iter_count = 0
                train_loss = []
                intervention_loss = []
                consistency_loss = []
                
                self.model.train()
                epoch_time = time.time()

                for i, batch_data in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()

                    if self.args.meta_dim > 0:  # With meta data
                        batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data = batch_data
                    else:  # Without meta data
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                        meta_data = None

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    if meta_data is not None:
                        meta_data = meta_data.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    binary_noncausal_mask = (self.noncausal_mask > 0)
                    apply_intervention_this_batch = torch.rand(1).item() < self.args.intervention_prob
                    if apply_intervention_this_batch:
                        noncausal_mask_for_batch = binary_noncausal_mask
                    else:
                        noncausal_mask_for_batch = None

                    # Forward pass
                    if 'PD' in self.args.model:
                        out = self.model(batch_x, batch_x_mark, meta_data=meta_data, noncausal_mask=noncausal_mask_for_batch) 
                    else:
                        out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    if self.args.use_intervention and apply_intervention_this_batch:
                        batch_y_target = batch_y[:, -self.args.pred_len:, :]
                        batch_y_expanded = batch_y_target.unsqueeze(0).expand(self.args.K, -1, -1, -1)
                        
                        mse_losses = F.mse_loss(out, batch_y_expanded, reduction='none')  # [K x bs x nvars x target_window]
                        weights = F.softmax(self.model.model.sample_weights, dim=0)  # [K]
                        weighted_loss = torch.sum(weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * mse_losses)
                        mean_pred = torch.mean(out, dim=0, keepdim=True)  # [1 x bs x nvars x target_window]
                        consistency_loss_val = F.mse_loss(out, mean_pred.expand(self.args.K, -1, -1, -1))
                        entropy_loss = -torch.sum(weights * torch.log(weights + 1e-10))
                        loss = weighted_loss + self.args.consistency_weight * consistency_loss_val + self.args.entropy_weight * entropy_loss
                    else:
                        out = out[:, -self.args.pred_len:, :] if out.dim() == 4 else out
                        loss = criterion(out, batch_y[:, -self.args.pred_len:, :])

                    train_loss.append(loss.item())
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                    
                    scheduler.step()

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.intervention_epochs - epoch) * len(train_loader) - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                
                avg_train_loss = np.average(train_loss)
                avg_intervention_loss = np.average(intervention_loss) if intervention_loss else 0
                avg_consistency_loss = np.average(consistency_loss) if consistency_loss else 0
                
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                print(f"Intervention Epoch {epoch + 1}: Train Loss: {avg_train_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
                
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
                if self.args.lradj != 'PD':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                else:
                    current_lr = model_optim.param_groups[0]['lr']
                    decay_factor = self.args.intervention_lr_decay
                    new_lr = current_lr * decay_factor
                    for param_group in model_optim.param_groups:
                        param_group['lr'] = new_lr
                    print('Updating learning rate to {:.6f}'.format(new_lr))

        return self.model

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        scaler = test_data.scaler

        print('loading model')
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        preds = []
        trues = []
        inputx = []
        total_loss = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                if self.args.meta_dim > 0:  # With meta data
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data = batch_data
                else:  # Without meta data
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    meta_data = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if meta_data is not None:
                    meta_data = meta_data.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'PD' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, meta_data=meta_data)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'PD' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, meta_data=meta_data)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                original_shape = batch_y.shape
                outputs = scaler.inverse_transform(outputs.reshape(-1, original_shape[-1]))
                batch_y = scaler.inverse_transform(batch_y.reshape(-1, original_shape[-1]))

                pred = outputs.reshape(original_shape) 
                true = batch_y.reshape(original_shape) 
                preds.append(pred)
                trues.append(true)
       
                batch_x = batch_x.detach().cpu().numpy()
                original_shape = batch_x.shape
                batch_x = scaler.inverse_transform(batch_x.reshape(-1, original_shape[-1]))
                batch_x = batch_x.reshape(original_shape)
                inputx.append(batch_x)

                if i % 20 == 0:
                    gt = np.concatenate((batch_x[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, f1 = metric(preds, trues)
        print('rmse:{}, mae:{}, f1:{}'.format(rmse, mae, f1))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}, mae:{}, f1:{}'.format(rmse, mae, f1))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'PD' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'PD' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
