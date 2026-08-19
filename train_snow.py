# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from snow_dataset import InMemoryH5
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia
import snow_config as C



'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 120 # total epoch
epoch_gap = 40  # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 16
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criteria_fusion = Fusionloss().to(device)
CDDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
CDDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    CDDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    CDDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
_SSIMCls = getattr(kornia.losses, 'SSIMLoss', None) or kornia.losses.SSIM
Loss_ssim = _SSIMCls(11, reduction='mean')


# data loader
trainloader = DataLoader(InMemoryH5(C.H5_PATH),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0, pin_memory=True)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

# ---- resume support -------------------------------------------------------
RESUME_PATH = os.path.join("models", "CDDFuse_snow_" + C.PAIR_NAME + "_last.pth")
os.makedirs("models", exist_ok=True)


def _save_ckpt(epoch):
    """Atomic write: temp file + rename, so a kill mid-save can't corrupt it."""
    state = {
        'epoch': epoch,
        'CDDF_Encoder': CDDF_Encoder.state_dict(),
        'CDDF_Decoder': CDDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
        'optimizer1': optimizer1.state_dict(), 'optimizer2': optimizer2.state_dict(),
        'optimizer3': optimizer3.state_dict(), 'optimizer4': optimizer4.state_dict(),
        'scheduler1': scheduler1.state_dict(), 'scheduler2': scheduler2.state_dict(),
        'scheduler3': scheduler3.state_dict(), 'scheduler4': scheduler4.state_dict(),
    }
    tmp = RESUME_PATH + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, RESUME_PATH)


start_epoch = 0
if os.path.exists(RESUME_PATH) and os.environ.get("SNOW_RESUME", "1") == "1":
    _ck = torch.load(RESUME_PATH, map_location=device)
    CDDF_Encoder.load_state_dict(_ck['CDDF_Encoder'])
    CDDF_Decoder.load_state_dict(_ck['CDDF_Decoder'])
    BaseFuseLayer.load_state_dict(_ck['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(_ck['DetailFuseLayer'])
    optimizer1.load_state_dict(_ck['optimizer1']); optimizer2.load_state_dict(_ck['optimizer2'])
    optimizer3.load_state_dict(_ck['optimizer3']); optimizer4.load_state_dict(_ck['optimizer4'])
    scheduler1.load_state_dict(_ck['scheduler1']); scheduler2.load_state_dict(_ck['scheduler2'])
    scheduler3.load_state_dict(_ck['scheduler3']); scheduler4.load_state_dict(_ck['scheduler4'])
    start_epoch = _ck['epoch'] + 1
    print("RESUMING from " + RESUME_PATH + " at epoch " + str(start_epoch))
else:
    print("starting fresh (no checkpoint at " + RESUME_PATH + ")")
# ---------------------------------------------------------------------------

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(start_epoch, num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        CDDF_Encoder.train()
        CDDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        CDDF_Encoder.zero_grad()
        CDDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap: #Phase I
            feature_V_B, feature_V_D, _ = CDDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = CDDF_Encoder(data_IR)
            data_VIS_hat, _ = CDDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = CDDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                CDDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                CDDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
        else:  #Phase II
            feature_V_B, feature_V_D, feature_V = CDDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = CDDF_Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            data_Fuse, feature_F = CDDF_Decoder(data_VIS, feature_F_B, feature_F_D)


            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                CDDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                CDDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6

    _save_ckpt(epoch)

if True:
    checkpoint = {
        'CDDF_Encoder': CDDF_Encoder.state_dict(),
        'CDDF_Decoder': CDDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "CDDFuse_snow_"+C.PAIR_NAME+"_"+timestamp+".pth")
    torch.save(checkpoint, out_path)
    print("\nSaved fusion weights ->", out_path)
