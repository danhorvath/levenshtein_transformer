import torch

# vector<vector<vector<uint32_t>>> suggested_ed2_path(
#     vector<vector<uint32_t>>& xs,
#     vector<vector<uint32_t>>& ys,
#     uint32_t terminal_symbol) {
#   vector<vector<vector<uint32_t>>> seq(xs.size());
#   for (uint32_t i = 0; i < xs.size(); i++) {
#     vector<vector<uint32_t>> d = edit_distance2_with_dp(xs.at(i), ys.at(i));
#     seq.at(i) =
#         edit_distance2_backtracking(d, xs.at(i), ys.at(i), terminal_symbol);
#   }
#   return seq;
# }


def suggested_ed2_path(xs, ys, terminal_symbol):
    seq = []
    for i, _ in enumerate(xs):
        distance = edit_distance2_with_dp(xs[i], ys[i])
        seq.append(edit_distance2_backtracking(
            distance, xs[i], ys[i], terminal_symbol))

    return seq


# vector < vector < uint32_t >> edit_distance2_with_dp(
#     vector < uint32_t > & x,
#     vector < uint32_t > & y) {
#     uint32_t l_x = x.size()
#     uint32_t l_y = y.size()
#     vector < vector < uint32_t >> d(l_x + 1, vector < uint32_t > (l_y + 1))
#     for (uint32_t i=0
#          i < l_x + 1
#          i++) {
#         d[i][0] = i
#     }
#     for (uint32_t j=0
#          j < l_y + 1
#          j++) {
#         d[0][j] = j
#     }
#     for (uint32_t i=1
#          i < l_x + 1
#          i++) {
#         for (uint32_t j=1
#              j < l_y + 1
#              j++) {
#             d[i][j] =
#             min(min(d[i - 1][j], d[i][j - 1]) + 1,
#                 d[i - 1][j - 1] + 2 * (x.at(i - 1) == y.at(j - 1) ? 0: 1))
#         }
#     }
#     return d
# }


def edit_distance2_with_dp(x, y):
    l_x = len(x)
    l_y = len(y)
    distance = [[0 for _ in range(l_y + 1)] for _ in range(l_x+1)]

    for i in range(l_x+1):
        distance[i][0] = i

    for j in range(l_y+1):
        distance[0][j] = j

    for i in range(1, l_x+1):
        for j in range(1, l_y+1):
            distance[i][j] = min(
                min(distance[i - 1][j], distance[i][j - 1]) + 1,
                distance[i - 1][j - 1] + 2 * (0 if x[i - 1] == y[j - 1] else 1)
            )
    return distance


# vector<vector<uint32_t>> edit_distance2_backtracking(
#     vector<vector<uint32_t>>& d,
#     vector<uint32_t>& x,
#     vector<uint32_t>& y,
#     uint32_t terminal_symbol) {
#   vector<uint32_t> seq;
#   vector<vector<uint32_t>> edit_seqs(x.size() + 2, vector<uint32_t>());
#   /*
#   edit_seqs:
#   0~x.size() cell is the insertion sequences
#   last cell is the delete sequence
#   */

#   if (x.size() == 0) {
#     edit_seqs.at(0) = y;
#     return edit_seqs;
#   }

#   uint32_t i = d.size() - 1;
#   uint32_t j = d.at(0).size() - 1;

#   while ((i >= 0) && (j >= 0)) {
#     if ((i == 0) && (j == 0)) {
#       break;
#     }

#     if ((j > 0) && (d.at(i).at(j - 1) < d.at(i).at(j))) {
#       seq.push_back(1); // insert
#       seq.push_back(y.at(j - 1));
#       j--;
#     } else if ((i > 0) && (d.at(i - 1).at(j) < d.at(i).at(j))) {
#       seq.push_back(2); // delete
#       seq.push_back(x.at(i - 1));
#       i--;
#     } else {
#       seq.push_back(3); // keep
#       seq.push_back(x.at(i - 1));
#       i--;
#       j--;
#     }
#   }

#   uint32_t prev_op, op, s, word;
#   prev_op = 0, s = 0;
#   for (uint32_t k = 0; k < seq.size() / 2; k++) {
#     op = seq.at(seq.size() - 2 * k - 2);
#     word = seq.at(seq.size() - 2 * k - 1);
#     if (prev_op != 1) {
#       s++;
#     }
#     if (op == 1) // insert
#     {
#       edit_seqs.at(s - 1).push_back(word);
#     } else if (op == 2) // delete
#     {
#       edit_seqs.at(x.size() + 1).push_back(1);
#     } else {
#       edit_seqs.at(x.size() + 1).push_back(0);
#     }

#     prev_op = op;
#   }

#   for (uint32_t k = 0; k < edit_seqs.size(); k++) {
#     if (edit_seqs[k].size() == 0) {
#       edit_seqs[k].push_back(terminal_symbol);
#     }
#   }
#   return edit_seqs;
# }

def edit_distance2_backtracking(distance, x, y, terminal_symbol):
    l_x = len(x)
    edit_seqs = [[] for _ in range(l_x+2)]
    seq = []

    if l_x == 0:
        edit_seqs[0] = y
        return edit_seqs

    i = len(distance)-1
    j = len(distance[0])-1

    while i >= 0 and j >= 0:
        if i == 0 and j == 0:
            break

        if j > 0 and distance[i][j-1] < distance[i][j]:
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

    xs = [[t for t in s if t != padding_idx]
          for i, s in enumerate(x_s.tolist())]

    ys = [[t for t in s if t != padding_idx]
          for i, s in enumerate(y_s.tolist())]

    print(f'{xs} => {ys}')
    print(suggested_ed2_path(xs, ys, padding_idx))
