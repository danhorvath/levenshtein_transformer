# This file is based on the fairseq implementation of the Levenshtein distance calculation.
# Their code was written in c, which I translated to python here.
import torch


def suggested_ed2_path(xs: list, ys: list, terminal_symbol: int):
    seq = []
    for i, _ in enumerate(xs):
        distance = edit_distance2_with_dp(xs[i], ys[i])
        seq.append(edit_distance2_backtracking(
            distance, xs[i], ys[i], terminal_symbol))

    return seq


def edit_distance2_with_dp(x: list, y: list):
    l_x = len(x)
    l_y = len(y)
    distance = [[0 for _ in range(l_y + 1)] for _ in range(l_x + 1)]

    for i in range(l_x + 1):
        distance[i][0] = i

    for j in range(l_y + 1):
        distance[0][j] = j

    for i in range(1, l_x + 1):
        for j in range(1, l_y + 1):
            distance[i][j] = min(
                min(distance[i - 1][j], distance[i][j - 1]) + 1,
                distance[i - 1][j - 1] + 2 * (0 if x[i - 1] == y[j - 1] else 1)
            )
    return distance


def edit_distance2_backtracking(distance, x: list, y: list, terminal_symbol: int):
    l_x = len(x)
    edit_seqs = [[] for _ in range(l_x + 2)]
    seq = []

    if l_x == 0:
        edit_seqs[0] = y
        return edit_seqs

    i = len(distance) - 1
    j = len(distance[0]) - 1

    while i >= 0 and j >= 0:
        if i == 0 and j == 0:
            break

        if j > 0 and distance[i][j - 1] < distance[i][j]:
            seq.append(1)  # insert
            seq.append(y[j - 1])
            j -= 1
        elif i > 0 and distance[i - 1][j] < distance[i][j]:
            seq.append(2)  # delete
            seq.append(x[i - 1])
            i -= 1
        else:
            seq.append(3)  # keep
            seq.append(x[i - 1])
            i -= 1
            j -= 1

    prev_op = 0
    s = 0
    l_s = len(seq)

    for k in range(l_s // 2):
        op = seq[l_s - 2 * k - 2]
        word = seq[l_s - 2 * k - 1]
        if prev_op != 1:
            s += 1
        if op == 1:  # insert
            edit_seqs[s - 1].append(word)
        elif op == 2:  # delete
            edit_seqs[l_x + 1].append(1)
        else:
            edit_seqs[l_x + 1].append(0)

        prev_op = op

    for _, edit_seq in enumerate(edit_seqs):
        if len(edit_seq) == 0:
            edit_seq.append(terminal_symbol)

    return edit_seqs


if __name__ == "__main__":
    padding_idx = 100

    x_s = torch.tensor([[1, 2, 3, 4, 5]])
    y_s = torch.tensor([[100, 100, 4, 3, 6, 5]])

    x_s = [[t for t in s if t != padding_idx] for i, s in enumerate(x_s.tolist())]
    y_s = [[t for t in s if t != padding_idx] for i, s in enumerate(y_s.tolist())]

    print(f'{x_s} => {y_s}')
    print(suggested_ed2_path(x_s, y_s, padding_idx))
