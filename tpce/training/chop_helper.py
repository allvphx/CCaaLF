import time

import numpy as np

N_TXN_TYPE = 4
IS_STORE_PROCEDURE = True
TXN_ACCESS_NUM = [30, 48, 10, 17]
TXN_ACCESS_START = [0, 0, 0, 0, 0]
for i in range(N_TXN_TYPE):
    for j in range(i+1, N_TXN_TYPE):
        TXN_ACCESS_START[j] += TXN_ACCESS_NUM[i]
# TXN_CRITICAL_ACCESS_NUM = [7, 3, 3]
print(TXN_ACCESS_START)
N_ACCESS = np.sum(TXN_ACCESS_NUM)
TXN_ACCESS_START[-1] = N_ACCESS
eps = 1e-5
TXN_ACCESS_ENDS = [TXN_ACCESS_START[i] + TXN_ACCESS_NUM[i] for i in range(N_TXN_TYPE)]

# There is no point for us to learn wait for rendezvous points.
LEARN_WAIT = [1 for _ in range(N_ACCESS)]
# Some accesses will never be separated during chop (e.g. loop stock read modify write)
SMALLEST_EXPOSE = [1 for _ in range(N_ACCESS)]


'''(deprecated) There are parameters used for leveled wait guard point learning.'''
# (deprecated) too many guard points tends to result in high spin wait cost, we limit it to 3.
MAX_N_GUARD_POINTS = 3
# (deprecated) we pick all critical guard points in IC3.
CRITICAL_GUARD_POINTS = [
    [0, 1, 3, 6, 7, 8, 9, 10, 11],
    [0, 2, 4, 6],
    [0, 2, 4, 6, 8]
]
# ---> encoding: for each transaction,
# we encode three ordered index to guard for three points respectively.
# (bishop mapping problem!) ---> we encode a path on a chess table C(n+m+1, n) !!!
# Some heuristic rules can be used during search.
# 1. if a transaction T1 has been guarded to x step of T2, it tends to guard to a higher point in latter.
# reason 1: cross wait tends to break the pipeline.
# reason 2: if we have waited to lower point to read the write from T2, T2's write should have also been read earlier.

N_CRITICAL_WAIT = LEARN_WAIT.count(1)
N_CRITICAL_EXPOSE = SMALLEST_EXPOSE.count(1)


def chop_domain_filter_encode(n, wait_chop, expose):
    assert len(wait_chop) == N_TXN_TYPE * n  # wait_chop now has length N_TXN_TYPE * n
    assert len(expose) == n  # expose has length n
    assert len(LEARN_WAIT) == n  # LEARN_WAIT and SMALLEST_EXPOSE remain tied to n
    assert len(SMALLEST_EXPOSE) == n
    res_wait_chop = []
    res_expose = []
    for i in range(n):
        if SMALLEST_EXPOSE[i] == 1:
            res_expose.append(expose[i])
    for i in range(N_TXN_TYPE * n):
        if LEARN_WAIT[i % n] == 1:
            res_wait_chop.append(wait_chop[i])

    assert len(res_expose) == N_CRITICAL_EXPOSE
    assert len(res_wait_chop) == N_CRITICAL_WAIT * N_TXN_TYPE

    return np.array(res_wait_chop), np.array(res_expose)


def chop_domain_filter_decode(n, encoded_wait_chop, encoded_expose):
    decoded_wait_chop = None
    decoded_expose = None
    wait_idx = 0
    expose_idx = 0

    if encoded_expose is not None:
        assert len(encoded_expose) == N_CRITICAL_EXPOSE
        decoded_expose = [0] * n
        for i in range(n):
            if SMALLEST_EXPOSE[i] == 1:
                decoded_expose[i] = encoded_expose[expose_idx]
                expose_idx += 1
            else:
                decoded_expose[i] = 0

    if encoded_wait_chop is not None:
        decoded_wait_chop = [0] * (N_TXN_TYPE * n)
        assert len(encoded_wait_chop) == N_CRITICAL_WAIT * N_TXN_TYPE
        for i in range(N_TXN_TYPE * n):
            if LEARN_WAIT[i % n] == 1:
                decoded_wait_chop[i] = encoded_wait_chop[wait_idx]
                wait_idx += 1
            else:
                decoded_wait_chop[i] = 0

        for i in range(N_TXN_TYPE):
            idx = 0
            for j in range(N_TXN_TYPE):
                for k in range(1, TXN_ACCESS_NUM[j]):
                    offset = i * n + idx + k
                    decoded_wait_chop[offset] = max(decoded_wait_chop[offset], decoded_wait_chop[offset - 1])
                idx += TXN_ACCESS_NUM[j]

    return np.array(decoded_wait_chop), np.array(decoded_expose)


def translate_wait_to_guard_points(wait, n_guard):
    # print(wait)
    txn_guard_info = np.zeros((N_TXN_TYPE, n_guard))
    encoded_wait = np.zeros((N_TXN_TYPE, N_TXN_TYPE, n_guard))
    # The jth transaction's wait on ith transaction.
    for i in range(N_TXN_TYPE):
        wait_segment = wait[i * N_CRITICAL_WAIT: (i + 1) * N_CRITICAL_WAIT]
        found_point = sorted(set(np.append(wait_segment, 0)))
        length = len(found_point)
        assert length <= n_guard + 1
        txn_guard_info[i][:length - 1] = found_point[1:]
        for j in range(length - 1, n_guard):
            txn_guard_info[i][j] = txn_guard_info[i][j - 1]
        for tx in range(N_TXN_TYPE):
            idx = 0
            for j in range(TXN_ACCESS_ENDS[tx - 1], TXN_ACCESS_ENDS[tx]):
                wait_value = wait_segment[j]
                while found_point[idx] < wait_value:
                    encoded_wait[i][tx][idx] = j - TXN_ACCESS_ENDS[tx - 1]
                    idx += 1
            while idx < n_guard:
                encoded_wait[i][tx][idx] = TXN_ACCESS_ENDS[tx] - TXN_ACCESS_ENDS[tx - 1]
                idx += 1
            assert idx == n_guard

    return encoded_wait, txn_guard_info


def reverse_translate_wait_to_guard_points(encoded_wait, txn_guard_info, n_guard):
    # Initialize the reconstructed wait array
    reconstructed_wait = np.zeros(N_TXN_TYPE * N_CRITICAL_WAIT)

    for i in range(N_TXN_TYPE):
        wait_segment = np.zeros(N_CRITICAL_WAIT)
        guard_points = [0] + list(txn_guard_info[i][:n_guard])

        for tx in range(N_TXN_TYPE):
            idx = 0
            access_start = TXN_ACCESS_ENDS[tx - 1]
            access_end = TXN_ACCESS_ENDS[tx]
            j = access_start

            while j < access_end:
                wait_value = guard_points[idx]  # Map from guard points
                access_idx = int(encoded_wait[i][tx][idx]) + access_start  # Recover access index
                # Set wait values between `access_start` and `access_idx`
                if access_idx > access_end:
                    # special case: all are zero.
                    for k in range(j, access_end):
                        wait_segment[k] = 0
                    break
                for k in range(j, access_idx):
                    wait_segment[k] = wait_value
                j = access_idx
                idx += 1
                if idx == n_guard:
                    for k in range(j, access_end):
                        wait_segment[k] = guard_points[-1]
                    break

        # Store the reconstructed wait segment
        reconstructed_wait[i * N_CRITICAL_WAIT: (i + 1) * N_CRITICAL_WAIT] = wait_segment

    return reconstructed_wait


def test_wait_access_encoder_decoder():
    np.random.seed(int(time.time()))
    for _ in range(10000):
        n_guard = np.random.randint(1, 5)  # Example guard points

        # Generate random wait values
        wait = None
        for i in range(N_TXN_TYPE):
            for tx in range(N_TXN_TYPE):
                rand_s = sorted(np.random.choice([0] + CRITICAL_GUARD_POINTS[i][:n_guard],
                                                 TXN_ACCESS_ENDS[tx] - TXN_ACCESS_ENDS[tx - 1]))
                if wait is not None:
                    wait = np.append(wait, rand_s)
                else:
                    wait = rand_s

        # Encode the wait values
        encoded_wait, txn_guard_info = translate_wait_to_guard_points(wait, n_guard)

        # Decode the wait values
        reconstructed_wait = reverse_translate_wait_to_guard_points(encoded_wait, txn_guard_info, n_guard)

        if not np.allclose(wait, reconstructed_wait):
            print("Original wait:", wait)
            print("Encoded wait:", encoded_wait)
            print("Transaction guard info:", txn_guard_info)
            print("Reconstructed wait:", reconstructed_wait)
            assert False, "Original and reconstructed wait arrays do not match!"
    print("Validation successful: Original and reconstructed wait arrays match!")


if __name__ == '__main__':
    # print(N_CRITICAL_WAIT * 3 + N_CRITICAL_EXPOSE)
    # print(N_CRITICAL_EXPOSE)
    # print(N_CRITICAL_WAIT)
    # a, b = translate_wait_to_guard_points(
    #     [0, 0, 1, 1, 4, 4, 4, 1, 4, 4, 1, 1, 1] + [0, 0, 1, 1, 4, 4, 7, 1, 4, 4, 4, 4, 7]
    #     + [0, 0, 2, 2, 4, 4, 4, 1, 4, 4, 4, 4, 4],
    #     3
    # )
    # print(reverse_translate_wait_to_guard_points(a, b, 3))
    test_wait_access_encoder_decoder()
