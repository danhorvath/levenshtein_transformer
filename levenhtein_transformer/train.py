import time
import wandb
from levenhtein_transformer.model import LevenshteinTransformerModel
from levenhtein_transformer.config import config


def run_epoch(data_iter, model: LevenshteinTransformerModel, criterion, opt, steps_so_far, batch_multiplier=1,
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

        ins_out = out["ins_out"]
        ins_tgt = out["ins_tgt"]
        ins_mask = out["ins_mask"]
        word_pred_out = out["word_pred_out"]
        word_pred_tgt = out["word_pred_tgt"]
        word_pred_mask = out["word_pred_mask"]
        word_del_out = out["word_del_out"]
        word_del_tgt = out["word_del_tgt"]
        word_del_mask = out["word_del_mask"]

        ins_loss = criterion(outputs=ins_out, targets=ins_tgt, masks=ins_mask)
        word_pred_loss = criterion(outputs=word_pred_out, targets=word_pred_tgt, masks=word_pred_mask)
        del_loss = criterion(outputs=word_del_out, targets=word_del_tgt, masks=word_del_mask)

        loss = ins_loss + word_pred_loss + del_loss
        if train:
            loss.backward()
            if optimizer_should_step:
                opt.optimizer.step()
                opt.optimizer.zero_grad()

        # TODO set number of batches if the number of iterations in the epoch is not dividable by batch_multiplier

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if logging and optimizer_should_step:
            elapsed = time.time() - start
            wandb.log({'Step': steps_so_far + effective_step,
                       'Loss': loss * batch_multiplier / batch.ntokens,
                       'Insertion loss': ins_loss * batch_multiplier / batch.ntokens,
                       'Word prediction loss': word_pred_loss * batch_multiplier / batch.ntokens,
                       'Deletion loss': del_loss * batch_multiplier / batch.ntokens,
                       'Tokens per sec': tokens / elapsed,
                       'Learning rate': opt._rate,
                       'Batch length': current_batch_size,
                       'Effective batch length': current_batch_size * config['batch_multiplier']})
            if effective_step % 100 == 1:
                print(f"Step: {steps_so_far + effective_step} | Loss: {loss * batch_multiplier / batch.ntokens} | " +
                      f"Insertion loss: {ins_loss * batch_multiplier / batch.ntokens} | " +
                      f"Word prediction loss: {word_pred_loss * batch_multiplier / batch.ntokens} | " +
                      f"Deletion loss: {del_loss * batch_multiplier / batch.ntokens} | " +
                      f"Tokens per Sec: {tokens / elapsed} | Learning rate: {opt._rate} | " +
                      f"Batch length: {current_batch_size}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens, effective_step
