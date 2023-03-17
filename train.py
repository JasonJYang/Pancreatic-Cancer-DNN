import os
import argparse
import collections
import pickle
import torch
import numpy as np
import pandas as pd
import model.loss as module_loss
import model.metric as module_metric
from data_loader.data_loaders import PCDataLoader as module_data
from model.model import MyDNN as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from model.metric import roc_auc


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    SEED = config['data_loader']['args']['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    config['data_loader']['args']['model_dir'] = config.save_dir
    data_loader = module_data(**config['data_loader']['args'])
    train_loader = data_loader.get_train_dataloader()
    valid_loader = data_loader.get_valid_dataloader()
    test_loader = data_loader.get_test_dataloader()
    feature_list = data_loader.get_feature_list()

    print('train: {}, valid: {}, test: {}'.format(len(train_loader.sampler),
        len(valid_loader.sampler), len(test_loader.sampler)))
    # print(data_loader.test_idx)

    feature_num = data_loader.get_feature_num()

    # build model architecture, then print to console
    model = module_arch(feature_num=feature_num,
                        dropout=config['arch']['args']['dropout'])
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      l1_decay=config['arch']['args']['l1_decay'],
                      data_loader=train_loader,
                      valid_data_loader=valid_loader,
                      test_data_loader=test_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    """Test."""
    logger = config.get_logger('test')
    logger.info(model)
    test_metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    def inference(infer_data_loader, display_string, save_file_name):
        logger.info(display_string)
        patient_list, y_pred_list, y_true_list = [], [], []
        with torch.no_grad():
            for batch_idx, (patient, data, target) in enumerate(infer_data_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)

                y_pred = torch.sigmoid(output)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = target.cpu().detach().numpy()
                
                patient_list.append(patient.numpy())
                y_pred_list.append(y_pred)
                y_true_list.append(y_true)
    
        patient = np.concatenate([patient_list[i] for i in range(len(patient_list))])
        y_pred = np.concatenate([y_pred_list[i] for i in range(len(y_pred_list))])
        y_true = np.concatenate([y_true_list[i] for i in range(len(y_true_list))])

        performance_dict = {'patient': patient, 'y_pred': y_pred, 'y_true': y_true}
        # with open(os.path.join(config.save_dir, save_file_name), 'wb') as f:
        #     pickle.dump(performance_dict, f)

        def result2df(state):
            with open(os.path.join(config.save_dir, state+'_patient2name_dict.pkl'), 'rb') as f:
                patient_index2name_dict = pickle.load(f)
            df = pd.DataFrame({'patient_index': performance_dict['patient'].tolist(),
                            'prediction': performance_dict['y_pred'].tolist(),
                            'true': performance_dict['y_true'].tolist()})
            df['patient'] = df['patient_index'].map(patient_index2name_dict)
            df.drop_duplicates(subset=['patient'], inplace=True)
            df.to_csv(os.path.join(config.save_dir, save_file_name), index=False)
            return df

        df = result2df(state=display_string)
        rocauc = roc_auc(y_pred=df['prediction'], y_true=df['true'])

        logger.info('ROC-AUC: {}'.format(rocauc))
    
    inference(train_loader, display_string='train', save_file_name='train_result.csv')
    inference(valid_loader, display_string='valid', save_file_name='valid_result.csv')
    inference(test_loader, display_string='test', save_file_name='test_result.csv')    

    logger.info('Captum to check the attributions of each variable...')
    trainer.captum_attributions(save_dir=config.save_dir, feature_list=feature_list)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
