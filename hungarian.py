import sys
import numpy as np
import random


def reduce_matrix(cost_matrix):
    """
    Reducerar radvis, sedan kolonnvis genom att subtrahera minimum.
    In: cost_matrix (np.array)
    Out: r_matrix (np.array)
    """
    r_matrix = np.copy(cost_matrix)
    r_dim = np.shape(r_matrix)[0]  # antal rader
    c_dim = np.shape(r_matrix)[1]  # antal kolumner
    if r_dim < c_dim:
        diff = c_dim - r_dim  # så många nya rader behövs
        new_rows = np.zeros((diff, c_dim))
        r_matrix = np.append(r_matrix, new_rows, axis=0)
    min_r = np.min(r_matrix, axis=1)

    for r in range(r_dim):
        r_matrix[r, :] -= min_r[r]
    min_c = np.min(r_matrix, axis=0)
    for c in range(c_dim):
        r_matrix[:, c] -= min_c[c]

    return r_matrix


def find_min_coord(bool_matrix):
    """
    Hittar den rad/kolumn där minsta antalet (>0) nollor finns, tillsammans med positionen på den raden
    där första nollan befinner sig.
    In: bool_matrix (np.array)
    Out: [row_i (int), col_i (int)]
    """
    r_dim, c_dim = np.shape(bool_matrix)
    zero_count_r = np.inf
    zero_count_c = np.inf
    row_i = np.inf
    for r in range(r_dim):
        row_count = 0
        for c in range(c_dim):
            if bool_matrix[r, c]:  # hittade en nolla
                row_count += 1
        if 0 < row_count < zero_count_r:  # hittade en ny bästa rad
            zero_count_r = row_count
            row_i = r
    col_i = np.where(bool_matrix[row_i])[0][0]
    # nu kollar vi om vi hittar en bättre kolumn än den rad vi har
    for c in range(c_dim):
        col_count = 0
        for r in range(r_dim):
            if bool_matrix[r, c]:
                col_count += 1
        if 0 < col_count < zero_count_c:
            zero_count_c = col_count
            if zero_count_c < zero_count_r:
                col_i = c
                row_i = np.where(bool_matrix[:, col_i])[0][0]

    return row_i, col_i


def get_bool_matrix(reduced_matrix):
    """
    Element är True om reduced_matrix har en nolla på den positionen, annars False.
    In: reduced_matrix (np.array)
    Out: bool_m (np.array)
    """
    bool_m = []
    r_dim, c_dim = np.shape(bool_matrix)
    for r in range(r_dim):
        bool_m_row = []
        for c in range(c_dim):
            if not reduced_matrix[r, c]:  # hittade en nolla
                bool_m_row.append(True)
            else:
                bool_m_row.append(False)
        bool_m.append(bool_m_row)
    return np.array(bool_m)


def get_marked_indexes(reduced_matrix):
    """
    Markerar den första nollan på den bästa tillgängliga raden/kolumnen så länge som det finns nollor i matrisen som
    ännu inte är besökta.
    In: reduced_matrix (np.array)
    Out: [marked (list), binary_m (np.array)]
    """
    r_dim, c_dim = np.shape(bool_matrix)
    bool_m = get_bool_matrix(reduced_matrix)
    binary_m = np.zeros((r_dim, c_dim), dtype=int)
    marked = []
    while np.count_nonzero(bool_m):  # så länge det finns minst en True i bool_m
        min_row_i, col_i = find_min_coord(bool_m)
        num_of_rows = 0
        num_of_cols = 0
        for k in range(r_dim):
            if bool_m[min_row_i, k]:
                num_of_rows += 1
            if bool_m[k, col_i]:
                num_of_cols += 1
        if num_of_rows > num_of_cols:  # rita horisontellt streck i binary_m
            binary_m[min_row_i, :] = np.ones(c_dim)
        else:
            binary_m[:, col_i] = np.ones(r_dim)
        marked.append([min_row_i, col_i])
        for j in range(r_dim):
            bool_m[min_row_i, j] = False
            bool_m[j, col_i] = False
    check_again = True
    uncovered = []
    for r in range(len(reduced_matrix)):
        for c in range(len(reduced_matrix)):
            if not binary_m[r, c]:
                uncovered += [reduced_matrix[r, c]]
    if uncovered:
        min_temp = min(x for x in uncovered if x > 0)
    else:
        min_temp = 0
    while check_again:
        check_again = False
        for r in range(r_dim):
            for c in range(c_dim):
                if not reduced_matrix[r, c] and not binary_m[r, c]:  # not covered 0 found
                    check_again = True
                    reduced_matrix[r, c] += min_temp
                    break
    return marked, binary_m


def is_at_intersection(binary_matrix, coord):
    return 0 not in binary_matrix[coord[0]] and 0 not in binary_matrix[:, coord[1]]


def shift_zeros(reduced_matrix, binary_matrix):
    not_covered = []
    r_dim = np.shape(binary_matrix)[0]
    c_dim = np.shape(binary_matrix)[1]
    for r in range(r_dim):
        for c in range(c_dim):
            if not binary_matrix[r, c]:  # inte täckt
                not_covered.append(reduced_matrix[r, c])  # lägg till det faktiska värdet
    min_val = min(not_covered)
    for r in range(r_dim):
        for c in range(c_dim):
            if not binary_matrix[r, c]:
                reduced_matrix[r, c] -= min_val
            if binary_matrix[r, c] and is_at_intersection(binary_matrix, (r, c)):
                reduced_matrix[r, c] += min_val
    return reduced_matrix


def unique_rows(marked):
    rows = []
    cols = []
    for row, col in marked:
        if row not in rows:
            rows += [row]
        if col not in cols:
            cols += [col]
    return min(len(rows), len(cols))  # will differ if we have multiple positions on the same line


def hungarian(matr):
    debug_list = []
    reduced_matrix = reduce_matrix(matr)
    total_cost = 0
    r_dim = len(reduced_matrix)
    orig_r = len(matr)
    orig_c = len(matr[0])
    lines = 0
    positions = []
    while lines < r_dim:
        positions, bin_m = get_marked_indexes(reduced_matrix)
        lines = unique_rows(positions)
        if lines < r_dim:
            # steg 4
            reduced_matrix = shift_zeros(reduced_matrix, bin_m)
    for pos in sorted(positions, key=lambda p: p[0]):
        if pos[0] < min(orig_r, orig_c):
            debug_list.append(pos[1])
            total_cost += matr[pos[0]][pos[1]]
    print(total_cost)
    print(debug_list)


mat = [[[random.randint(0, 10) for x in range(50)] for y in range(50)]for _ in range(100)]
for i, m in enumerate(mat):
    print(i)
    hungarian(m)
