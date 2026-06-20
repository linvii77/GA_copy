"""Standalone test script: load two best checkpoints and run final evaluation."""
import logging
import sys
import numpy as np
import torch
from networks.vnet import VNet
from utils import test_util_vnet_AB

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── config ────────────────────────────────────────────────────────────────────
CKPT_DIR  = './model/Synapse_CPS_fused_e2_GA_4labeled_seed_1337'
CKPT_A    = f'{CKPT_DIR}/iter_11500_dice_0.663776_best_A.pth'
CKPT_B    = f'{CKPT_DIR}/iter_11500_dice_0.663776_best_B.pth'
BASE_DIR  = './data/Synapse/'
TEST_LIST = ['0004', '0007', '0010', '0033', '0035', '0036']
PATCH     = (96, 96, 96)
NUM_CLS   = 14
ORGAN_NAMES = [
    'spleen', 'r.kidney', 'l.kidney', 'gallbladder', 'esophagus',
    'liver', 'stomach', 'aorta', 'ivc',
    'portal and splenic vein', 'pancreas',
    'right adrenal gland', 'Left adrenal gland',
]
# ─────────────────────────────────────────────────────────────────────────────

def create_model(n_classes):
    model = VNet(n_channels=1, n_classes=n_classes).cuda()
    return model

logging.info(f'Loading A: {CKPT_A}')
model_A = create_model(NUM_CLS)
model_A.load_state_dict(torch.load(CKPT_A))
model_A.eval()

logging.info(f'Loading B: {CKPT_B}')
model_B = create_model(NUM_CLS)
model_B.load_state_dict(torch.load(CKPT_B))
model_B.eval()

logging.info('Running full test on 6 test cases...')
_, _, metric_final = test_util_vnet_AB.validation_all_case(
    model_A, model_B,
    num_classes=NUM_CLS,
    base_dir=BASE_DIR,
    image_list=TEST_LIST,
    patch_size=PATCH,
    stride_xy=32,
    stride_z=16,
    data_format='h5',
)

# metric_final: [N_cases, 4, N_organs]  — matches training script convention
# metric_mean[m][o] = metric m, organ o
mean = np.mean(metric_final, axis=0)   # [4, N_organs]
std  = np.std(metric_final, axis=0)    # [4, N_organs]

logging.info(
    f'Final Average  DSC:{mean[0].mean():.4f}  HD95:{mean[1].mean():.4f}'
    f'  NSD:{mean[2].mean():.4f}  ASD:{mean[3].mean():.4f}'
)
for i, name in enumerate(ORGAN_NAMES):
    logging.info(
        f'{name:30s}  DSC:{mean[0,i]:.4f}±{std[0,i]:.4f}'
        f'  HD95:{mean[1,i]:.4f}±{std[1,i]:.4f}'
        f'  NSD:{mean[2,i]:.4f}±{std[2,i]:.4f}'
        f'  ASD:{mean[3,i]:.4f}±{std[3,i]:.4f}'
    )
