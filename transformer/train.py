import time
import wandb
from transformer.config import config


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
        optimizer_should_step = effective_step.is_integer()

        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

        # TODO set number of batches if the number of iterations in the epoch is not dividable by batch_multiplier
        loss = loss_compute(out, batch.trg_y, batch.ntokens, optimizer_step=optimizer_should_step)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        current_batch_size = max(batch.src.size(0) * batch.src.size(1), batch.trg.size(0) * batch.trg.size(1))

        if logging and optimizer_should_step:
            elapsed = time.time() - start
            wandb.log({'Step': steps_so_far + effective_step, 'Loss': loss * batch_multiplier / batch.ntokens,
                       'Tokens per sec': tokens / elapsed, 'Learning rate': loss_compute.opt._rate,
                       'batch_length': current_batch_size,
                       'effective_batch_length': current_batch_size * config['batch_multiplier']})
            if effective_step % 100 == 1:
                print(f"Step: {steps_so_far + effective_step} | Loss: {loss * batch_multiplier / batch.ntokens} " +
                      f"| Tokens per Sec: {tokens / elapsed} | Learning rate: {loss_compute.opt._rate} | Batch length: {current_batch_size}")
            start = time.time()
            tokens = 0

    return (total_loss / total_tokens, effective_step)
