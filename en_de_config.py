BATCH_MULTIPLIER = 2.
BATCH_SIZE = 26000

config = {'max_epochs': 100, # max_step will terminate the training
          'num_layers': 6,
          'batch_size': BATCH_SIZE,
          'batch_multiplier': BATCH_MULTIPLIER,
          'effective_batch_size':BATCH_SIZE * BATCH_MULTIPLIER,
          'max_len': 150,
          'min_freq': 1,
          'warmup_init_lr': 1e-07,
          'warmup_end_lr': 7e-4,
          'warmup': 4000,
          'epsilon': 1e-9,
          'max_step': 1e5,
          'beta_1': 0.9,
          'beta_2': 0.98
          }