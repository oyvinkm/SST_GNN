
import os
import enlighten
import torch
import copy
import numpy as np
from datetime import datetime
from loguru import logger
from torch import nn
from matplotlib import pyplot as plt





class LOGMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.log(self.mse(input, target))

class Trainer(object):
  def __init__(self, beta = 0.001, criterion = None, device = 'cpu'):
      self.time = datetime.now().strftime("%Y_%m_%d-%H.%M")
      self.beta = beta
      self.device = device
      self.model_path = self.log_model_path()
      self.image_folder = self.log_image_path()
      self.plot_path = self.log_plot_path()
      if criterion == 'MSE':
         self.criterion = nn.MSELoss()
      else:
        self.criterion = LOGMSELoss()

  def log_model_path(self):
    model_name = f"model_VAE_{self.time}"
    folder = 'model_chkpoints'
    if not os.path.exists(folder):
      os.mkdir(folder)
      logger.info(f'Created folder: {folder}')
    model_path = os.path.join(folder, model_name + ".pt")
    logger.success(f'Model path: {model_path}')
    return model_path

  def log_image_path(self):
    folder = 'image_chkpoints'
    if not os.path.exists(folder):
      os.mkdir(folder)
      logger.info(f'Created folder: {folder}')
    image_path = os.path.join(folder, f'images_{self.time}')
    if not os.path.exists(image_path):
       os.mkdir(image_path)
       logger.info(f'Created folder: {image_path}')
    return image_path
  
  def log_plot_path(self):
    folder = 'plots'
    if not os.path.exists(folder):
      os.mkdir(folder)
      logger.info(f'Created folder: {folder}')
    plot_path = os.path.join(folder, f'plot_{self.time}')
    return plot_path
  
  def save_plot(self, test_loss, val_loss):
    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(1,2, 1)
    plt.plot(test_loss)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.add_subplot(1,2,2)
    plt.plot(val_loss)
    plt.title('Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(f'{self.plot_path}.png')
    plt.close()



  def save_image(self, pred, true, epoch):
    fig = plt.figure(figsize = (12, 8))
    fig.add_subplot(2, 1, 1)
    plt.imshow((pred.permute(1,2,0).cpu().detach().numpy()))
    plt.axis('off')
    plt.title('Prediction')
    fig.add_subplot(2, 1, 2)
    plt.imshow((true.permute(1,2,0).cpu().detach().numpy()))
    plt.axis('off')
    plt.title('Prediction')
    fig.savefig(os.path.join(self.image_folder, f'viz_{epoch}'))
    plt.close()

  def save_model(self, epoch, net_state_dict, opt_state_dict):
    torch.save({
            'epoch': epoch,
            'state_dict': net_state_dict,
            'optimizer_state_dict': opt_state_dict,
            }, self.model_path)

  def validate(self, net, val_loader, epoch):
    val_loss = []
    with torch.no_grad():
      for j, (data, _) in enumerate(val_loader):
          inputs = data.to(self.device)
          logger.debug(f'Val input: {inputs.shape}')
          _, recon = net(inputs, Train = False)
          rec_loss = self.criterion(recon, inputs)
          val_loss.append(rec_loss.item())
          if epoch % 10 == 0 and j == 0: 
            self.save_image(recon[0], inputs[0], epoch)
    return sum(val_loss)/len(val_loss)

  def train(self, no_epochs, 
            net, 
            augmentation, 
            train_loader, 
            val_loader, 
            optimizer,
            device = 'cpu',
            progress_bar = True):
    epoch_losses = []
    val_loss = []
    if progress_bar:
      manager = enlighten.get_manager()
      epochs = manager.counter(total=no_epochs, desc="Epochs", unit="Epochs", color="red")
    best_val_loss = np.inf
    tot_val_loss = np.inf
    for epoch in range(no_epochs):  # Loop over the dataset multiple times
        epochs.update()
        batch_counter = manager.counter(total=len(train_loader), desc="Batches", unit="Batches", color="blue", leave=False, position=True)
        best_model = None
        losses = []
        kls = []
        for i, (data, _) in enumerate(train_loader):
            batch_counter.update()
            # Get the inputs; data is a list of [inputs, labels]
            inputs = augmentation(data.to(self.device))
            # Zero the parameter gradients

            # Forward + backward + optimize
            kl, recon = net(inputs)
            kls.append(kl.cpu().detach().numpy())
            rec_loss = self.criterion(recon, inputs)
            loss = self.beta*kl + rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss.item()
            losses.append(l)
        val_loss = []
        print('VALIDATING...')
        
        tot_val_loss = self.validate(net, val_loader, epoch)
        val_loss.append(tot_val_loss)
        if tot_val_loss < best_val_loss:
            logger.info('Saving model')
            best_model = copy.deepcopy(net)
            self.save_model(epoch, net.state_dict(), optimizer.state_dict())
        batch_counter.close()
        epoch_losses.append(np.mean(losses))
        #kulback = np.mean(kls)
        logger.info('[%d, %5d] train loss: %.3f\tKL Divergence: %.3f\tKL: %.3f\tval loss: %.3f' %
              (epoch + 1, i + 1, epoch_losses[-1], self.beta*np.mean(kls), np.mean(kls), tot_val_loss))
    epochs.close()
    manager.stop() 
    self.save_plot(epoch_losses, val_loss)   
    return best_model, epoch_losses
  
  def test(self, net, test_loader):
    test_loss = []
    with torch.no_grad():
      for i, (data, _) in enumerate(test_loader):
        inputs = data.to(self.device)
        _, recon = net(inputs, Train = False)
        rec_loss = self.criterion(recon, inputs)
        test_loss.append(rec_loss.item())
        self.save_image(recon[0], inputs[0], f'test_{i}')
    return sum(test_loss)/len(test_loss)