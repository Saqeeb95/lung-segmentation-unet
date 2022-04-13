import torch
from torch import optim
from tqdm import tqdm
from utils.dice_score import dice_loss, validation_dice_coeff
from evaluate import evaluate
import torch.nn as nn
import torch.nn.functional as F

def train_net(net, train_ds, train_dl, valid_dl, batch_size, optimizer=None, lr_scheduler=None, loss_func=None,
              val_percent=0.1, epochs: int = 5, lr: float = 0.001, weight_decay=0,
              device=torch.device('cpu'), save_checkpoint: bool = True, amp: bool = False):

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, amp=amp))

    # logging.info(f'''Starting training:
    #     Epochs:          {epochs}
    #     Batch size:      {batch_size}
    #     Learning rate:   {learning_rate}
    #     Training size:   {n_train}
    #     Validation size: {n_val}
    #     Checkpoints:     {save_checkpoint}
    #     Device:          {device.type}
    #     Images scaling:  {img_scale}
    #     Mixed Precision: {amp}
    # ''')

    # # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # if optimizer is None:
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # if lr_scheduler is None:
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()

    optimizer = optimizer
    scheduler = lr_scheduler
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    global_step = 0

    n_train = len(train_ds)

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for images, true_masks in train_dl:
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                # true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.to(device=device)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    # loss = criterion(masks_pred, true_masks) \
                    #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(true_masks.long(), net.n_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=False)
                    # loss = criterion(masks_pred, true_masks) \
                    #         + dice_loss(F.softmax(masks_pred, dim=1).float(), true_masks)
                    if loss_func == "CrossEntropyLoss":
                        loss = criterion(masks_pred, true_masks)
                    elif loss_func == "DiceLoss":
                        loss = dice_loss(F.softmax(masks_pred, dim=1).float(), true_masks)
                    elif loss_func == "CrossEntropy+DiceLoss" or loss_func is None:
                        loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(), true_masks)
                    

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, valid_dl, device)
                        scheduler.step(val_score)

    net.eval()
    with torch.no_grad():
        dice = validation_dice_coeff(net, valid_dl)
    return dice.item()

        #                 logging.info('Validation Dice score: {}'.format(val_score))
        #                 experiment.log({
        #                     'learning rate': optimizer.param_groups[0]['lr'],
        #                     'validation Dice': val_score,
        #                     'images': wandb.Image(images[0].cpu()),
        #                     'masks': {
        #                         'true': wandb.Image(true_masks[0].float().cpu()),
        #                         'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
        #                     },
        #                     'step': global_step,
        #                     'epoch': epoch,
        #                     **histograms
        #                 })

        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
        #     logging.info(f'Checkpoint {epoch + 1} saved!')