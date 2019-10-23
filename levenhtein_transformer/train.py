import time
import wandb
from levenhtein_transformer.layers import LevenshteinEncodeDecoder
from levenhtein_transformer.config import config


def run_epoch(data_iter, model: LevenshteinEncodeDecoder, opt, steps_so_far, batch_multiplier=1,
              logging=False, train=False):
    """
    Standard Training and Logging Function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    effective_step = 0

    for i, batch in enumerate(data_iter):
        effective_step = i / batch_multiplier

        # update the model on steps defined by batch_multiplier or the last step in the epoch
        optimizer_should_step = effective_step.is_integer()
        current_batch_size = max(batch.src.size(0) * batch.src.size(1), batch.trg.size(0) * batch.trg.size(1))

        out = model(batch.src, batch.noised_trg, batch.src_mask, batch.noised_trg_mask, batch.trg)

        ins_loss = out['ins_loss'].sum().item()
        word_pred_loss = out['word_pred_loss'].sum().item()
        word_del_loss = out['word_del_loss'].sum().item()

        loss = out['loss'].sum()
        if train:
            loss.backward()
            if optimizer_should_step:
                opt.step()
                opt.optimizer.zero_grad()

        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if logging and optimizer_should_step:
            elapsed = time.time() - start
            wandb.log({'Step': steps_so_far + effective_step,
                       'Loss': loss,
                       'Insertion loss': ins_loss,
                       'Word prediction loss': word_pred_loss,
                       'Deletion loss': word_del_loss,
                       'Tokens per sec': tokens / elapsed,
                       'Learning rate': opt._rate,
                       'Batch length': current_batch_size,
                       'Effective batch length': current_batch_size * config['batch_multiplier']})
            if effective_step % 100 == 1:
                print(f"Step: {steps_so_far + effective_step} | Loss: {loss} | " +
                      f"Insertion loss: {ins_loss} | " +
                      f"Word prediction loss: {word_pred_loss} | " +
                      f"Deletion loss: {word_del_loss} | " +
                      f"Tokens per Sec: {tokens / elapsed} | Learning rate: {opt._rate} | " +
                      f"Batch length: {current_batch_size}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens, effective_step
