from torch.cuda import device_count

BATCH_MULTIPLIER = 2.
SINGLE_GPU_BATCH_SIZE = 7000
BATCH_SIZE = SINGLE_GPU_BATCH_SIZE * device_count()

config = {'max_epochs': 100,  # max_step will terminate the training
          'num_layers': 6,
          'attn_heads': 8,
          'ff_dim': 2048,
          'model_dim': 512,
          'single_gpu_real_batch_size': SINGLE_GPU_BATCH_SIZE,
          'batch_size': BATCH_SIZE,
          'batch_multiplier': BATCH_MULTIPLIER,
          'effective_batch_size': BATCH_SIZE * BATCH_MULTIPLIER,
          'val_batch_size': SINGLE_GPU_BATCH_SIZE * BATCH_MULTIPLIER // 2,
          'max_decode_iter': 10,
          'max_len': 150,
          'min_freq': 1,
          'warmup_init_lr': 1e-07,
          'warmup_end_lr': 0.0005,
          'min_lr': 1e-09,
          'warmup': 10000,
          'dropout': 0.0,
          'decoder_dropout': 0.1,
          'weight_decay': 0.01,
          'epsilon': 1e-9,
          'max_step': 3e5,
          'beta_1': 0.9,
          'beta_2': 0.98,
          'max_decoder_ratio': 2
          }
