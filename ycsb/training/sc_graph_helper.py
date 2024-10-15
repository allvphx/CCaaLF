import numpy as np

from chop_helper import *

# According to PolyJuice's chopping, all accesses are:


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
    # access_t[5] = expose_t[5] = 0
    # access_t[6] = expose_t[6] = 0
    # access_t[9] = expose_t[9] = 0
    # access_t[10] = expose_t[10] = 0
    print("size = ", N_ACCESS)
    access_t = np.array([1 for i in range(N_ACCESS)])
    expose_t = SMALLEST_EXPOSE
    print("conflict detection:")
    print("".join([str(i) for i in access_t]))
    print("wait priorities:")
    print(" ".join(['0' for _ in range(N_ACCESS)]))
    print("timeout:")
    print(" ".join(["100000" for _ in range(N_ACCESS)]))
    print("expose:")
    print("".join([str(i) for i in expose_t]))
    tmp = calculate_wait_access(expose_t,
                                access_t)
    print("chop wait:")
    print(" ".join([str(int(v)) for v in tmp]))
    print("extra:")
    print("32 2 2 2 2 2 2")

    print("starting point for PolyJuice ---------")
    print("txn buffer size\n32\n"
          "txn commit backoff part, 3 lines for retry times [0, 1, >=2] (decrease backoff)\n2\n2\n2\n"
          "txn abort backoff part, 3 lines for retry times [0, 1, >=2] (increase backoff)\n2\n2\n2")
    print("txn abort backoff part, 3 lines for retry times [0, 1, >=2] (increase backoff)")
    for i in range(N_ACCESS):
        print("1 1 %d %d" % (expose_t[i], tmp[i]))
    for i in range(20 - N_ACCESS):
        print("0 0 0 0")

