import sys
import numpy as np


def reduce_matrix(cost_matrix):
    """
    Reducerar radvis, sedan kolonnvis genom att subtrahera minimum.
    In: cost_matrix (np.array)
    Out: r_matrix (np.array)
    """
    r_matrix = np.copy(cost_matrix)
    r_dim = np.shape(r_matrix)[0]  # antal rader
    c_dim = np.shape(r_matrix)[1]  # antal kolumner
    min_r = np.min(r_matrix, axis=1)

    for r in range(r_dim):
        r_matrix[r, :] -= min_r[r]
    min_c = np.min(r_matrix, axis=0)
    for c in range(c_dim):
        r_matrix[:, c] -= min_c[c]

    if r_dim < c_dim:
        diff = c_dim - r_dim  # så många nya rader behövs
        new_rows = np.zeros((diff, c_dim))
        r_matrix = np.append(r_matrix, new_rows, axis=0)

    return r_matrix


def find_min_coord(bool_matrix):
    """
    Hittar den rad/kolumn där minsta antalet (>0) nollor finns, tillsammans med positionen på den raden
    där första nollan befinner sig.
    In: bool_matrix (np.array)
    Out: [row_i (int), col_i (int)]
    """
    r_dim = np.shape(bool_matrix)[0]
    c_dim = np.shape(bool_matrix)[1]
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
    r_dim = np.shape(reduced_matrix)[0]
    c_dim = np.shape(reduced_matrix)[1]
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

    r_dim = np.shape(reduced_matrix)[0]
    c_dim = np.shape(reduced_matrix)[1]
    bool_m = get_bool_matrix(reduced_matrix)
    binary_m = np.zeros((r_dim, c_dim), dtype=int)
    marked = []
    while np.count_nonzero(bool_m):  # så länge det finns minst en True i bool_m
        min_row_i, col_i = find_min_coord(bool_m)
        num_of_rows = 0
        num_of_cols = 0
        for i in range(r_dim):
            if bool_m[min_row_i, i]:
                num_of_rows += 1
            if bool_m[i, col_i]:
                num_of_cols += 1

        if num_of_rows > num_of_cols:  # rita horisontellt streck i binary_m
            binary_m[min_row_i] = np.ones(c_dim)
        else:
            binary_m[:, col_i] = np.ones(r_dim)

        marked.append([min_row_i, col_i])
        for i in range(r_dim):
            bool_m[min_row_i, i] = False
            bool_m[i, col_i] = False

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


def hungarian():
    """
    Hämtar dimension och matris från input-fil.
    [python3 hungarian.py < INPUTFILE]
    """
    debug_list = []
    r_dim = int(sys.stdin.readline())

    inp = []
    for i in range(r_dim):
        i_r = []
        snigel = str(sys.stdin.readline())
        for j in range(len(snigel)):
            if snigel[j] != '\n' and snigel[j] != ' ':

                x = int(snigel[j])
                i_r.append(x)
        inp.append(i_r)

    test_cost = np.array(inp)

    # testmatriser
    # test_cost_m = np.matrix([[1, 2, 3, 4], [5, 2, 9, 6], [1, 2, 2, 4], [8, 3, 7, 2]])
    # kattis_easy = np.array([[2, 1, 2, 2], [1, 2, 2, 2], [2, 2, 1, 2], [2, 2, 2, 1]])
    # kattis_hard = np.array([[5, 1, 1], [5, 4, 9], [6, 2, 5]])
    # kattis_hard2 = np.array([[5, 5, 6], [1, 4, 2], [1, 9, 5]])
    # ha_test = np.array([[82, 83, 69, 92], [77, 37, 49, 92], [11, 69, 5, 86], [8, 9, 98, 23]])
    # test_rand = np.array(np.random.random_integers(0, 10, (12, 12)))

    # cost_matrix = kattis_easy
    # cost_matrix = kattis_hard2
    # cost_matrix = ha_test
    cost_matrix = test_cost
    reduced_matrix = reduce_matrix(cost_matrix)
    total_cost = 0

    lines = 0
    positions = []
    while lines < r_dim:

        positions, bin_m = get_marked_indexes(reduced_matrix)
        lines = len(positions)
        if lines < r_dim:
            # steg 4
            reduced_matrix = shift_zeros(reduced_matrix, bin_m)

    for pos in sorted(positions, key=lambda p: p[0]):
        if pos[1] < r_dim:
            debug_list.append(pos[1]+1)
            total_cost += cost_matrix[pos[0], pos[1]]

    i = 0
    outp = ""
    while i < len(debug_list)-1:
        outp += str(debug_list[i]) + " "
        i += 1
    outp += str(debug_list[-1]) + "\n"
    print(total_cost)
    print(outp)


hungarian()
