import os
import pickle
import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import roc_auc
from utils.captum_visualize import attributions_visulaize


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_fns, optimizer, config,
                 data_loader, valid_data_loader=None, test_data_loader=None,
                 l1_decay=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.l1_decay = l1_decay

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (patient, data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            l1_loss = self._l1_loss_compute()
            loss = self.criterion(output, target) + l1_loss
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.train_metrics.update(met.__name__, met(y_pred, y_true))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_'+k : v for k, v in val_log.items()}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, save_flag=False):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        patient_list, y_pred_list, y_true_list = [], [], []
        with torch.no_grad():
            for batch_idx, (patient, data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                l1_loss = self._l1_loss_compute()
                loss = self.criterion(output, target) + l1_loss

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(y_pred, y_true))

                patient_list.append(patient.numpy())
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        if save_flag:
            validation_dict = {'patient': patient_list, 
                            'y_pred': y_pred_list,
                            'y_true': y_true_list}
            with open(os.path.join(self.config.save_dir, 'validation_dict.pkl'), 'wb') as f:
                pickle.dump(validation_dict, f)

        return self.valid_metrics.result()

    def captum_attributions(self, save_dir, feature_list):
        self.model.eval()
        with torch.no_grad():
            X_test_list = []
            for batch_idx, (patient, data, target) in enumerate(self.valid_data_loader):
                X_test_list.append(data)
            X_train_list = []
            for batch_idx, (patient, data, target) in enumerate(self.data_loader):
                X_train_list.append(data)    
        X_test = torch.cat(X_test_list).to(self.device)
        X_train = torch.cat(X_train_list).to(self.device)
        attributions_visulaize(self.model, X_test, X_train, 
                               os.path.join(save_dir, 'attribution.csv'), 
                               feature_list)

    def valid(self):
        self.model.eval()
        total_loss = 0.0
        patient_list, y_pred_list, y_true_list = [], [], []
        with torch.no_grad():
            for batch_idx, (patient, data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                l1_loss = self._l1_loss_compute()
                loss = self.criterion(output, target) + l1_loss

                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                
                patient_list.append(patient.numpy())
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)

        
        patient = np.concatenate([patient_list[i] for i in range(len(patient_list))])
        y_pred = np.concatenate([y_pred_list[i] for i in range(len(y_pred_list))])
        y_true = np.concatenate([y_true_list[i] for i in range(len(y_true_list))])
        valid_output = {'rocauc': roc_auc(y_pred, y_true)}

        valid_dict = {'patient': patient, 'y_pred': y_pred, 'y_true': y_true}
        with open(os.path.join(self.config.save_dir, 'valid_dict.pkl'), 'wb') as f:
            pickle.dump(valid_dict, f)

        return valid_output


    def test(self):
        self.model.eval()
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_fns))
        patient_list, y_pred_list, y_true_list = [], [], []
        with torch.no_grad():
            for batch_idx, (patient, data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                l1_loss = self._l1_loss_compute()
                loss = self.criterion(output, target) + l1_loss

                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size

                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                
                patient_list.append(patient.numpy())
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)

                for i, metric in enumerate(self.metric_fns):
                    total_metrics[i] += metric(y_pred, y_true) * batch_size
        
        test_output = {'n_samples': len(self.test_data_loader.sampler),
                       'total_loss': total_loss,
                       'total_metrics': total_metrics}
        
        patient = np.concatenate([patient_list[i] for i in range(len(patient_list))])
        y_pred = np.concatenate([y_pred_list[i] for i in range(len(y_pred_list))])
        y_true = np.concatenate([y_true_list[i] for i in range(len(y_true_list))])

        test_dict = {'patient': patient, 
                     'y_pred': y_pred,
                     'y_true': y_true}
        with open(os.path.join(self.config.save_dir, 'test_dict.pkl'), 'wb') as f:
            pickle.dump(test_dict, f)

        return test_output

    def _l1_loss_compute(self):
        l1_regularization = 0
        for param in self.model.parameters():
            l1_regularization += torch.sum(abs(param))
        return l1_regularization * self.l1_decay

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
