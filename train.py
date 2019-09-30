import time
import wandb
from en_de_config import config


def run_epoch(data_iter, model, loss_compute, steps_so_far, batch_multiplier=1, logging=False):
    """
    Standard Training and Logging Function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        effective_step = i / batch_multiplier

        # update the model on steps defined by batch_multiplier or the last step in the epoch
        optimizer_step = effective_step.is_integer()

        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)

        # TODO set number of batches if the number of iterations in the epoch is not dividable by batch_multiplier
        loss = loss_compute(out, batch.trg_y, batch.ntokens,
                            optimizer_step=optimizer_step)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if logging and optimizer_step:
            elapsed = time.time() - start
            wandb.log({'Step': steps_so_far + effective_step, 'Loss': loss * batch_multiplier / batch.ntokens,
                       'Tokens per Sec': tokens / elapsed, 'Learning rate': loss_compute.opt._rate,
                       'batch_length': len(batch.src), 'effective_batch_length': len(batch.src)*config['batch_multiplier']})
            if effective_step % 100 == 1:
                print(f"Step: {steps_so_far + effective_step} | Loss: {loss * batch_multiplier / batch.ntokens} " +
                      f"| Tokens per Sec: {tokens / elapsed} | Learning rate: {loss_compute.opt._rate} | Batch length: {len(batch.src)}")
            start = time.time()
            tokens = 0

    return (total_loss / total_tokens, effective_step)
