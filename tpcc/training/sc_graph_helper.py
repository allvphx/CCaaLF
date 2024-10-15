import numpy as np

from chop_helper import *

# According to PolyJuice's chopping, all accesses are:
access = [
    # New Order's accesses
    {"operation": "read", "table": "warehouse"},  # access_id 0
    {"operation": "read", "table": "district"},  # access_id 1
    {"operation": "write", "table": "district"},  # access_id 2
    {"operation": "read", "table": "item"},  # access_id 3
    {"operation": "read", "table": "stock"},  # access_id 4
    {"operation": "write", "table": "stock"},  # access_id 5
    {"operation": "write", "table": "new order"},  # access_id 6 - insert new order
    {"operation": "write", "table": "order"},  # access_id 7 - insert order
    {"operation": "write", "table": "order index"},  # access_id 8 - insert order index
    {"operation": "write", "table": "order line"},  # access_id 9 - insert order line
    {"operation": "read", "table": "customer"},  # access_id 10

    # Payment's accesses
    {"operation": "read", "table": "warehouse"},  # access_id 11
    {"operation": "write", "table": "warehouse"},  # access_id 12
    {"operation": "read", "table": "district"},  # access_id 13
    {"operation": "write", "table": "district"},  # access_id 14
    {"operation": "read", "table": "customer"},  # access_id 15
    {"operation": "write", "table": "customer"},  # access_id 16
    {"operation": "write", "table": "history"},  # access_id 17 - insert history

    # Delivery's accesses
    {"operation": "read", "table": "new order"},  # access_id 18 - scan new order table
    {"operation": "write", "table": "new order"},  # access_id 19 - delete new order
    {"operation": "read", "table": "order"},  # access_id 20
    {"operation": "write", "table": "order"},  # access_id 21
    {"operation": "read", "table": "order line"},  # access_id 22 - scan order line
    {"operation": "write", "table": "order line"},  # access_id 23
    {"operation": "read", "table": "customer"},  # access_id 24
    {"operation": "write", "table": "customer"}  # access_id 25
]

rend_cap = [11, 6, 8]


def get_full_conflict_graph(access_t=None):
    # transactions inner conflict by their steps.
    assert len(access) == N_ACCESS
    inner_conflict = np.zeros((N_ACCESS, N_ACCESS), dtype=np.int)
    access_idx = 0
    for i in range(N_TXN_TYPE):
        for j in range(TXN_ACCESS_NUM[i]):
            if j > 0:
                inner_conflict[access_idx, access_idx] = 1
            access_idx += 1

    # two separate transactions conflict on conflict operations.
    cross_conflict = np.zeros((N_ACCESS, N_ACCESS), dtype=int)
    for i in range(N_ACCESS):
        for j in range(N_ACCESS):
            if access[j]['table'] == access[i]['table'] and (
                    'write' == access[i]['operation'] or 'write' == access[j]['operation']):
                cross_conflict[i, j] = cross_conflict[j, i] = 1

    if access_t is not None:
        # Special case: when using occ for an operation, we assume no conflict for it!
        # case 1: for write occ, then there would not exist self write conflict and likely not exist dirty read on it.
        # case 2: for read occ, the dirty writer does not need to be blocked. (no conflict)
        for i in range(N_ACCESS):
            if access[i]['operation'] == 'write' and access_t[i] == 0:
                cross_conflict[i] = 0
                cross_conflict[:, i] = 0
            if access[i]['operation'] == 'read' and access_t[i] == 0:
                cross_conflict[i] = 0
                cross_conflict[:, i] = 0

    return cross_conflict


def one_step(old, graph):
    n = len(graph)
    res = old.copy()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i, j] += old[j, k] * graph[k, j]
    return res


def transit_closure(graph):
    n = len(graph)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                graph[i, j] |= graph[j, k] & graph[k, j]
    return graph


def get_wait_access_info(g, expose, access_t):
    n = len(g)
    wait_access = []
    assert n == len(expose)

    for i in range(N_TXN_TYPE):
        wait_to = np.zeros(N_ACCESS)
        piece_conflict = np.zeros(n, dtype=int)
        last_expose = TXN_ACCESS_NUM[i]
        for j in reversed(range(TXN_ACCESS_START[i], TXN_ACCESS_START[i + 1])):
            if expose[j]:
                last_expose = j
            if access[j]['operation'] == 'read':
                # For read, wait to this point.
                wait_to[(piece_conflict == 0) & (g[j] == 1)] = j - TXN_ACCESS_START[i] + 1
            else:
                # For write, wait to the expose point.
                wait_to[(piece_conflict == 0) & (g[j] == 1)] = last_expose - TXN_ACCESS_START[i] + 1
            piece_conflict = np.bitwise_or(piece_conflict, g[j])
            # if expose[j]:
            #     piece_conflict = np.zeros(n, dtype=int)
        # occ + wait policy is totally covered by dirty read/write + wait.
        wait_to[access_t == 0] = 0
        climbed_wait = np.zeros(N_ACCESS, dtype=int)
        # print(wait_to)
        for j in range(N_TXN_TYPE):
            expose_wait_point = 0
            for k in range(TXN_ACCESS_START[j], TXN_ACCESS_START[j + 1]):
                if k > 0 and expose[k - 1]:
                    climbed_wait[k] = expose_wait_point
                    expose_wait_point = 0
                if access[k]['operation'] == 'read':
                    climbed_wait[k] = max(climbed_wait[k], wait_to[k])
                else:
                    expose_wait_point = max(expose_wait_point, wait_to[k])
            climbed_wait[TXN_ACCESS_START[j]] = 0  # there is no reason for use to wait during first operation.
            mask = climbed_wait[TXN_ACCESS_START[j]: TXN_ACCESS_START[j + 1]] > rend_cap[i]
            climbed_wait[TXN_ACCESS_START[j]: TXN_ACCESS_START[j + 1]][mask] = rend_cap[i]  # Rendezvous Piece
        wait_access.extend(climbed_wait)
        # print("\t".join([str(int(v)) for v in climbed_wait]))

    rendezvous = rend_cap.copy()
    # Rendezvous analysis to tighten the cap.
    for i in range(N_TXN_TYPE):
        bar = rendezvous[i]
        while bar > 0 and access_t[bar + TXN_ACCESS_START[i] - 1] == 0:
            bar -= 1
        rendezvous[i] = bar
        for j in range(i * N_ACCESS, (i+1) * N_ACCESS):
            wait_access[j] = min(wait_access[j], rendezvous[i])

    # print(rendezvous)

    for i in range(N_ACCESS):
        # Final pass, fill case 3 in chop wait.
        has_wait = False
        for j in range(N_TXN_TYPE):
            if wait_access[i + j * N_ACCESS] > 0:
                has_wait = True
        if has_wait:
            for j in range(N_TXN_TYPE):
                if wait_access[i + j * N_ACCESS] == 0:
                    wait_access[i + j * N_ACCESS] = rendezvous[j]
    # print("result", wait_access)
    return wait_access


def calculate_wait_access(expose, access_t):
    c_graph = get_full_conflict_graph(access_t)
    return get_wait_access_info(c_graph, expose, access_t)


if __name__ == '__main__':
    # tmp = calculate_wait_access([int(bit) for bit in "10110000000010101101000000"],
    #                             [int(bit) for bit in "11100000000111110010000000"])
    # print(" ".join([str(int(v)) for v in tmp]))
    tmp = calculate_wait_access([int(bit) for bit in "10100000000010100000000000"],
                                [int(bit) for bit in "11110000000111110010000110"])
    print(" ".join([str(int(v)) for v in tmp]))
    # tmp = calculate_wait_access(SMALLEST_EXPOSE,
    #                             np.array([1 for i in range(N_ACCESS)]))
    # print(" ".join([str(int(v)) for v in tmp]))
