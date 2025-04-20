import argparse, logging, time, torch
from architectures import LDM_Segmentor
from torch.amp import autocast, GradScaler
from torch.nn.functional import l1_loss
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from config import *
from dataset import get_dataloaders

def trainer(model, optimizer, train_loader, device, scaler, val_loader):
    train_losses = []
    val_losses = []
    for epoch in range(100):
        print(f'starting epoch {epoch + 1}.')
        epoch_loss = 0
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            image, mask = batch['aug_image'], batch['aug_mask']

            with autocast(device, enabled=True):
                pred = model(image, mask)
                loss = l1_loss(pred, image)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        print(f'epoch {epoch + 1} train_loss: {train_losses[-1]}')
        if (epoch + 1) % VAL_INTERVAL == 0:
            val_losses.append(validator(model, val_loader, device))
            print(f'epoch {epoch + 1} val_loss: {val_losses[-1]}')
        
        print(f'ending epoch {epoch + 1}.')
        
    return train_losses, val_losses

def validator(model, val_loader, device):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            image, mask = batch['aug_image'], batch['aug_mask']

            with autocast(device, enabled=True):
                pred = model(image, mask)
                loss = l1_loss(pred, image)
                
            epoch_loss += loss.item()

    return epoch_loss / len(val_loader)

def main():
    print('initializing components.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)

    model = LDM_Segmentor().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0001)
    scaler = GradScaler(device)
    train_loader = get_dataloaders(
                BASE_DIR, split_ratio = SPLIT_RATIOS, split = 'train',
                trainsize = TRAINSIZE, batch_size = BATCH_SIZE, format = FORMAT
            )
    val_loader   = get_dataloaders(
        BASE_DIR, split_ratio = SPLIT_RATIOS, split = 'val',
        trainsize = TRAINSIZE, batch_size = BATCH_SIZE, format = FORMAT
    )

    print('starting training.')
    start_time   = time.time()
    losses = trainer(model, optimizer, train_loader, device, scaler, val_loader)      
    print(f'execution time: {(time.time() - start_time) // 60.0} minutes')

if __name__ == '__main__':
    main()