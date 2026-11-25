import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool
import os
import time
import leidenalg as la
import igraph as ig
from collections import defaultdict, Counter, deque
import random
import math
import psutil
import pickle
import xmeasures

_SKLEARN_STATE = {}

os.environ.setdefault("NUMBA_CACHE_DIR", "E:/HIDC-UD/.numba_cache")

MERGE_DUMP_DIR = None
MERGE_DUMP_TAG = 'ds'

_process = psutil.Process(os.getpid())

def get_memory_mb():
    return _process.memory_info().rss / (1024 * 1024)

def memory_snapshot(label=""):
    mb = get_memory_mb()
    tag = f"  [{label}]" if label else ""
    print(f"[MEM{tag}] {mb:.2f} MB")
    return mb

class _IdentityIndex(object):
    __slots__ = ('_n',)
    def __init__(self, n): self._n = int(n)
    def __getitem__(self, k):
        if k < 0 or k >= self._n:
            raise IndexError(k)
        return int(k)
    def get(self, k, default=None):
        return int(k) if 0 <= k < self._n else default
    def __contains__(self, k): return 0 <= k < self._n
    def __len__(self): return self._n

def _dedup_undirected_edges(u, v, n_nodes):
    lo = np.minimum(u, v).astype(np.int64, copy=False)
    hi = np.maximum(u, v).astype(np.int64, copy=False)
    key = lo * np.int64(n_nodes) + hi

    uk, first_pos, inv = np.unique(key, return_inverse=True, return_index=True)
    inv = np.asarray(inv).ravel()
    w = np.bincount(inv, minlength=uk.size).astype(np.float64)
    seq = np.argsort(first_pos, kind='stable')
    del first_pos, inv, key, lo, hi
    return (uk // n_nodes)[seq], (uk % n_nodes)[seq], w[seq]

def _csr_from_dedup(iu, iv, w, n_nodes):
    rows = np.concatenate([iu, iv])
    cols = np.concatenate([iv, iu])
    ws = np.concatenate([w, w])
    ne = iu.size

    ekey = np.tile(np.arange(ne, dtype=np.int64), 2)
    order = np.lexsort((ekey, rows))
    cols = cols[order]
    ws = ws[order]
    row_len = np.bincount(rows, minlength=n_nodes).astype(np.int64)
    del rows, ekey, order
    return row_len, cols, ws

def _dense_comm_array(comm_dict, n_graph):
    n_d = len(comm_dict)
    nodes_arr = np.fromiter(comm_dict.keys(), dtype=np.int64, count=n_d)
    lab_arr = np.fromiter(comm_dict.values(), dtype=np.int64, count=n_d)
    uniq_lab, dense = np.unique(lab_arr, return_inverse=True)
    dense = np.asarray(dense).ravel()
    comm_dense = np.full(int(n_graph), -1, dtype=np.int64)
    comm_dense[nodes_arr] = dense
    return uniq_lab, dense, comm_dense

def _comm_pair_weights(ei, ej, ew, ci, cj, K):
    mask = (ci >= 0) & (cj >= 0) & (ci != cj)
    if not mask.any():
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=np.float64)
    lo = np.minimum(ci[mask], cj[mask]).astype(np.int64)
    hi = np.maximum(ci[mask], cj[mask]).astype(np.int64)
    key = lo * np.int64(K) + hi
    ukey, inv = np.unique(key, return_inverse=True)
    inv = np.asarray(inv).ravel()
    wsum = np.bincount(inv, weights=ew[mask], minlength=ukey.size)
    return ukey // K, ukey % K, wsum

HIDDEN_DIM = 16
EPOCHS = 20
NUM_PROCESSES = max(1, min(6, (os.cpu_count() or 2)))
LEIDEN_ITERATIONS = 20
TEMP_DIR = os.environ.get("ASNCD_TEMP_DIR", "temp_embeddings")
RANDOM_SEED = int(os.environ.get("ASNCD_SEED", "42"))
ALPHA_2S = 0.8
MIN_BLOCK_SIZE=10

CN_ALPHA_MAX = 2.0
ENABLE_MB    = True

ENABLE_SGM   = True
SE_COLD_START = False

ENABLE_CPM_CLEANUP = True

FUSE_OUTER = 0
ENABLE_MB_FALLBACK = True

MB_FB_SCORE  = 0.1
MB_FB_MAX_FRAC = 0.01
MB_FB_DENS_GUARD = 0.1

ENABLE_SG_SPLIT = True
ENABLE_ORPHAN_MERGE = True

MIN_COMM_SIZE_OVERRIDE = None

ENABLE_LPA_CANDIDATE = True
LPA_CAND_MIN_N       = 10000

LPA_CAND_MAX_DEG     = 8.0
LPA_CAND_RUNS        = 3

LPA_CAND_SEED_BASE   = 1000 + max(0, RANDOM_SEED - 42)

LPA_CAND_MAX_GROWTH = 6

LPA_CAND_MODE            = 'refine'

LPA_CAND_SHAPE_GUARD     = True

BLOCK_SIZE_OVERRIDE    = None

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import numpy as np
import networkx as nx

import math

def get_adaptive_params(N, E):

    BLOCKS_PER_CORE = 3
    num_workers = 5

    SPARSE_DEGREE = 1
    DENSE_DEGREE = 30
    SPARSE_FACTOR = 0
    DENSE_FACTOR = 2

    avg_deg = 2 * E / N

    if avg_deg < SPARSE_DEGREE:
        density = SPARSE_FACTOR
    elif avg_deg > DENSE_DEGREE:
        density = DENSE_FACTOR
    else:
        density = SPARSE_FACTOR + (DENSE_FACTOR - SPARSE_FACTOR) / (DENSE_DEGREE - SPARSE_DEGREE) * (avg_deg - SPARSE_DEGREE)

    max_by_parallel = N // (num_workers * BLOCKS_PER_CORE)

    block = int(max_by_parallel * density)
    block_size = max(1000, min(block, 50000))

    d_critical = 6
    base_tau = max(2, int(math.log10(N)))

    if avg_deg < d_critical:
        tau = max(2, base_tau - 1)
    else:
        tau = base_tau

    k_min, k_max = 5, 200
    alpha_max, alpha_min = CN_ALPHA_MAX, 0.0

    if avg_deg <= k_min:
        alpha = alpha_max
    elif avg_deg >= k_max:
        alpha = alpha_min
    else:

        log_ratio = (math.log(avg_deg) - math.log(k_min)) / \
                    (math.log(k_max) - math.log(k_min))

        alpha = alpha_max + (alpha_min - alpha_max) * (log_ratio ** 0.8)

    alpha = round(alpha, 2)

    return block_size, tau, alpha

RS_CORE_REF = 0.05
RS_CORE_K = 10

RS_HALFWIDTH_DEC = 0.5
RS_MIN = 10.0 ** (-RS_HALFWIDTH_DEC)
RS_MAX = 10.0 ** (+RS_HALFWIDTH_DEC)

RS_SHAPE = "sqrt"

def compute_structure_signals(n_nodes, edge_df):
    try:
        u = edge_df['u'].values.astype(np.int64)
        v = edge_df['v'].values.astype(np.int64)
        g = ig.Graph(n=int(n_nodes),
                     edges=list(zip(u.tolist(), v.tolist())),
                     directed=False)
        g.simplify()
        lcc_frac = max(len(c) for c in g.connected_components()) / max(n_nodes, 1)
        frac10core = float(np.mean(np.asarray(g.coreness()) >= RS_CORE_K))
        return frac10core, float(lcc_frac)
    except Exception as e:

        print(f"  [RS-adaptive] structure signals failed ({e}), fallback (0.08, 1.0)")
        return 0.08, 1.0

def get_adaptive_resolution_scale(frac10core, lcc_frac):

    rs_lo = 10.0 ** (-RS_HALFWIDTH_DEC)
    rs_hi = 10.0 ** (+RS_HALFWIDTH_DEC)
    s = min(max(frac10core / RS_CORE_REF, 0.0), 1.0)
    if RS_SHAPE == "linear":
        g = s
    elif RS_SHAPE == "log":
        g = math.log2(1.0 + s)
    else:
        g = math.sqrt(s)
    rs = rs_lo + (rs_hi - rs_lo) * g
    phi = min(max(lcc_frac, 0.0), 1.0)
    rs = phi * rs + (1.0 - phi) * rs_lo
    return round(rs, 3)

def split_data_by_metis_optimized(edge_df, all_nodes, node_degree_dict, block_size):
    import metis
    from collections import defaultdict

    n = len(all_nodes)
    num_parts = max(2, n // block_size)

    if isinstance(edge_df, pd.DataFrame):
        u_arr = edge_df['u'].values
        v_arr = edge_df['v'].values
    else:
        u_arr = np.asarray(edge_df)[:, 0]
        v_arr = np.asarray(edge_df)[:, 1]

    _node_arr = np.fromiter(all_nodes, dtype=np.int64, count=n)
    identity = bool(n and np.array_equal(_node_arr, np.arange(n)))

    parts = None
    if identity:
        try:
            import ctypes
            uu = np.asarray(u_arr, dtype=np.int64)
            vv = np.asarray(v_arr, dtype=np.int64)
            m_e = uu.size

            src = np.concatenate([uu, vv])
            dst = np.concatenate([vv, uu])
            rowi = np.tile(np.arange(m_e, dtype=np.int64), 2)
            order = np.lexsort((rowi, src))
            adjncy = dst[order].astype(np.int32)
            del order, rowi, src, dst
            row_len = np.bincount(np.concatenate([uu, vv]), minlength=n)
            xadj = np.zeros(n + 1, dtype=np.int32)
            xadj[1:] = np.cumsum(row_len).astype(np.int32)
            del row_len
            idx_t = metis.idx_t
            xadj_c = (ctypes.c_int32 * (n + 1)).from_buffer(xadj)
            adjncy_c = (ctypes.c_int32 * adjncy.size).from_buffer(adjncy)
            _mg = metis.METIS_Graph(idx_t(n), idx_t(1), xadj_c, adjncy_c,
                                    None, None, None)
            edgecuts, parts = metis.part_graph(_mg, nparts=num_parts,
                                               recursive=True)
        except Exception as e:
            print(f"  [METIS] CSR path failed ({e}), fallback to list adjacency")
            parts = None

    if parts is None:
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        idx_to_node = {i: node for i in node in enumerate(all_nodes)} if False \
            else {i: node for i, node in enumerate(all_nodes)}
        adjacency = [[] for _ in range(n)]
        for u, v in edge_df[['u', 'v']].values if isinstance(edge_df, pd.DataFrame) \
                else edge_df:
            i, j = node_to_idx[u], node_to_idx[v]
            adjacency[i].append(j)
            adjacency[j].append(i)
        try:
            edgecuts, parts = metis.part_graph(adjacency, nparts=num_parts,
                                               recursive=True)
        except Exception as e:
            print(f"  [METIS] recursive failed ({e}), trying k-way...")
            edgecuts, parts = metis.part_graph(adjacency, nparts=num_parts,
                                               recursive=False)

    parts = np.asarray(parts, dtype=np.int64)
    if identity:
        pu = parts[np.asarray(u_arr, dtype=np.int64)]
        pv = parts[np.asarray(v_arr, dtype=np.int64)]
    else:
        pu = parts[[node_to_idx[x] for x in u_arr]]
        pv = parts[[node_to_idx[x] for x in v_arr]]
    cut = int(np.count_nonzero(pu != pv))
    total = int(pu.size)
    del pu, pv

    sort_order = np.argsort(parts, kind='stable')
    uniq_parts, first_pos, counts = np.unique(parts, return_index=True,
                                             return_counts=True)
    seg_off = np.concatenate(([0], np.cumsum(counts)))
    seq = np.argsort(first_pos)
    blocks = [_node_arr[sort_order[seg_off[r]:seg_off[r + 1]]].tolist()
              for r in seq]

    print(f"  [METIS] {len(blocks)} blocks, cut edges: {cut}/{total} ({cut/total:.1%})")

    return blocks, all_nodes

def split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_size):
    visited = set()
    adjacency = defaultdict(list)

    for u, v in edge_df[['u', 'v']].values:
        adjacency[u].append(v)
        adjacency[v].append(u)

    sorted_nodes = sorted(all_nodes, key=lambda x: node_degree_dict.get(x, 0), reverse=True)

    blocks = []
    current_block = []

    for node in sorted_nodes:
        if node in visited:
            continue

        queue = deque([node])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            current_block.append(current)

            if len(current_block) >= block_size:
                blocks.append(current_block)
                current_block = []

            neighbors = adjacency.get(current, [])
            neighbors_sorted = sorted(neighbors, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
            for neighbor in neighbors_sorted:
                queue.append(neighbor)

    if current_block:
        if len(current_block) <= block_size and len(blocks) > 0:
            if len(blocks[-1]) + len(current_block) <= block_size * 1.5:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        else:
            blocks.append(current_block)

    unvisited_nodes = [node for node in all_nodes if node not in visited]
    for i in range(0, len(unvisited_nodes), block_size):
        small_block = unvisited_nodes[i:i+block_size]
        if small_block:
            blocks.append(small_block)

    new_all_nodes = []
    for block in blocks:
        new_all_nodes.extend(block)
    new_all_nodes = list(set(new_all_nodes))

    return blocks, new_all_nodes

try:
    from numba import njit as _njit_cn

    @_njit_cn(cache=True)
    def _common_neighbors_njit(eu, ev, indptr, cols):
        m = eu.shape[0]
        out = np.empty(m, dtype=np.int64)
        for e in range(m):
            u = eu[e]
            v = ev[e]
            i = indptr[u]
            i_end = indptr[u + 1]
            j = indptr[v]
            j_end = indptr[v + 1]
            common = 0
            while i < i_end and j < j_end:
                a = cols[i]
                b = cols[j]
                if a == b:
                    common += 1
                    i += 1
                    j += 1
                elif a < b:
                    i += 1
                else:
                    j += 1
            out[e] = common
        return out

    _CN_NUMBA_OK = True
except Exception:
    _CN_NUMBA_OK = False

def calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha,
                                    node_degree_dict=None):
    try:
        start_time = time.time()
        block_node_set = set(block_nodes)

        block_mask = edge_df['u'].isin(block_node_set) & edge_df['v'].isin(block_node_set)
        block_edge = edge_df[block_mask].copy()
        edge_count = len(block_edge)

        if edge_count == 0:
            return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))

        block_edge[['u_sorted', 'v_sorted']] = np.sort(block_edge[['u', 'v']].values, axis=1)
        edge_counts = block_edge.groupby(['u_sorted', 'v_sorted']).size().reset_index(name='count')

        if _CN_NUMBA_OK:
            _eu = edge_counts['u_sorted'].values.astype(np.int64)
            _ev = edge_counts['v_sorted'].values.astype(np.int64)
            _m = _eu.shape[0]
            _nodes_all = np.concatenate((_eu, _ev))
            _uniq, _inv = np.unique(_nodes_all, return_inverse=True)
            _k = _uniq.shape[0]
            _inv = _inv.astype(np.int64)
            _rows = np.concatenate((_inv[:_m], _inv[_m:]))
            _cols = np.concatenate((_inv[_m:], _inv[:_m]))
            _order = np.lexsort((_cols, _rows))
            _cols_sorted = _cols[_order]
            _indptr = np.zeros(_k + 1, dtype=np.int64)
            np.cumsum(np.bincount(_rows, minlength=_k), out=_indptr[1:])
            edge_counts['common'] = _common_neighbors_njit(
                _inv[:_m], _inv[_m:], _indptr, _cols_sorted)
        else:
            u_sorted_arr = edge_counts['u_sorted'].tolist()
            v_sorted_arr = edge_counts['v_sorted'].tolist()

            neighbor_dict = defaultdict(list)
            for u, v in zip(u_sorted_arr, v_sorted_arr):
                neighbor_dict[u].append(v)
                neighbor_dict[v].append(u)

            for u in neighbor_dict:
                neighbor_dict[u].sort()

            def count_common(u, v):
                neighbors_u = neighbor_dict.get(u, [])
                neighbors_v = neighbor_dict.get(v, [])
                i = j = common = 0
                len_u, len_v = len(neighbors_u), len(neighbors_v)

                while i < len_u and j < len_v:
                    if neighbors_u[i] == neighbors_v[j]:
                        common += 1
                        i += 1
                        j += 1
                    elif neighbors_u[i] < neighbors_v[j]:
                        i += 1
                    else:
                        j += 1
                return common

            edge_counts['common'] = [count_common(u, v)
                                    for u, v in zip(u_sorted_arr, v_sorted_arr)]

        deg_u_arr = edge_counts['u_sorted'].map(node_degree_dict).fillna(1).values.astype(np.float64)
        deg_v_arr = edge_counts['v_sorted'].map(node_degree_dict).fillna(1).values.astype(np.float64)
        deg_product = deg_u_arr * deg_v_arr

        scale_factor = 1.0 / (1.0 + np.log1p(deg_product))

        base_weight = edge_counts['count'].values + cn_base_alpha * edge_counts['common'].values
        edge_counts['weight'] = base_weight * scale_factor

        result_edge = block_edge[['u', 'v']].drop_duplicates()
        result_edge = result_edge.merge(
            edge_counts[['u_sorted', 'v_sorted', 'weight']],
            left_on=['u', 'v'],
            right_on=['u_sorted', 'v_sorted'],
            how='left'
        ).fillna(1)[['u', 'v', 'weight']]

        result_edge['weight'] = result_edge['weight'].clip(lower=0.001)

        return (block_id, result_edge)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))

def generate_block_community(args):
    try:
        weighted_edge_block, block_nodes, block_id, resolution_scale = args
        start_time = time.time()

        block_G = nx.Graph()
        if not weighted_edge_block.empty:
            valid_edges = weighted_edge_block[['u', 'v', 'weight']].values
            if len(valid_edges) > 0:

                block_G.add_weighted_edges_from(valid_edges.tolist())

        for node in block_nodes:
            if node not in block_G:
                block_G.add_node(node)

        if not block_G.nodes():
            return (block_id, {})

        node_list = sorted(block_G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        block_ig = ig.Graph(directed=False)
        block_ig.add_vertices(len(node_list))

        edges, edge_weights = [], []
        for u, v, data in block_G.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            edge_weights.append(float(data.get('weight', 1.0)))

        if edges:
            block_ig.add_edges(edges)
            block_ig.es['weight'] = edge_weights

        n = block_ig.vcount()
        m = block_ig.ecount()
        if m > 0 and n > 1:
            total_w = sum(edge_weights)

            weighted_density = total_w / (n * (n - 1) / 2)
            avg_w = total_w / m

            resolution = resolution_scale * np.sqrt(avg_w * weighted_density)

            resolution = max(0.001, min(resolution, 20.0))
        else:
            resolution = 1.0

        n_iterations = 30
        partition = la.find_partition(
            block_ig,
            la.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=42
        )

        leiden_comm = np.array(partition.membership)
        block_fine_comm = [leiden_comm[node_to_idx[node]] if node in node_to_idx else 0
                          for node in block_nodes]

        global_comm_prefix = block_id * 1000000
        block_comm_dict = {node: global_comm_prefix + comm
                          for node, comm in zip(block_nodes, block_fine_comm)}

        return (block_id, block_comm_dict)

    except Exception as e:
        block_comm_dict = {node: block_id * 1000000 + i for i, node in enumerate(block_nodes)}
        return (block_id, block_comm_dict)

class _EdgeWeightsView(object):
    __slots__ = ('_iu', '_iv', '_w', '_sk', '_spos')
    def __init__(self, iu, iv, w):
        self._iu = iu; self._iv = iv; self._w = w
        self._sk = None; self._spos = None
    def __len__(self):
        return int(self._w.size)
    def __iter__(self):
        return zip(self._iu.tolist(), self._iv.tolist())
    def keys(self):
        return zip(self._iu.tolist(), self._iv.tolist())
    def values(self):
        return iter(self._w.tolist())
    def items(self):
        return zip(zip(self._iu.tolist(), self._iv.tolist()), self._w.tolist())
    def get(self, key, default=None):
        i, j = int(key[0]), int(key[1])
        if i > j:
            i, j = j, i
        if self._sk is None:
            if self._iu.size == 0:
                return default
            sk = (self._iu.astype(np.int64) << 32) | self._iv.astype(np.int64)
            order = np.argsort(sk, kind='stable')
            self._sk = sk[order]
            self._spos = order
        target = (np.int64(i) << 32) | np.int64(j)
        pos = int(np.searchsorted(self._sk, target))
        if pos < self._sk.size and self._sk[pos] == target:
            return float(self._w[int(self._spos[pos])])
        return default

class _CSRAdj(object):
    __slots__ = ('_off', '_cols', '_ws', '_n')
    def __init__(self, off, cols, ws, n):
        self._off = off; self._cols = cols; self._ws = ws; self._n = int(n)
    def __getitem__(self, idx):
        cs = int(self._off[idx]); ce = int(self._off[idx + 1])
        return list(zip(self._cols[cs:ce].tolist(), self._ws[cs:ce].tolist()))
    def get(self, idx, default=None):
        if 0 <= idx < self._n:
            return self[idx]
        return default
    def __contains__(self, idx):
        return 0 <= idx < self._n
    def __len__(self):
        return self._n

def build_lightweight_graph(edge_df, nodes):
    if isinstance(edge_df, pd.DataFrame):
        u_raw = edge_df['u'].values
        v_raw = edge_df['v'].values
    else:
        _arr = np.asarray(edge_df)
        u_raw = _arr[:, 0]
        v_raw = _arr[:, 1]
    u_raw = np.asarray(u_raw, dtype=np.int64)
    v_raw = np.asarray(v_raw, dtype=np.int64)

    if nodes is not None:
        _node_arr = np.fromiter(nodes, dtype=np.int64, count=len(nodes))
        identity = bool(_node_arr.size and
                        np.array_equal(_node_arr, np.arange(_node_arr.size)))
    else:
        _node_arr = None
        identity = False

    if identity:

        n_nodes = int(_node_arr.size)
        i_idx = u_raw
        j_idx = v_raw
        node_to_idx = _IdentityIndex(n_nodes)
        idx_to_node = _IdentityIndex(n_nodes)
    else:

        if _node_arr is None:
            _node_arr = np.unique(np.concatenate([u_raw, v_raw]))
        else:
            _node_arr = np.sort(_node_arr)
        n_nodes = int(_node_arr.size)
        i_idx = np.searchsorted(_node_arr, u_raw)
        j_idx = np.searchsorted(_node_arr, v_raw)
        _nl = _node_arr.tolist()
        node_to_idx = {n: i for i, n in enumerate(_nl)}
        idx_to_node = {i: n for i, n in enumerate(_nl)}

    ei, ej, ew = _dedup_undirected_edges(i_idx, j_idx, n_nodes)
    row_len, csr_cols, csr_ws = _csr_from_dedup(ei, ej, ew, n_nodes)
    del i_idx, j_idx
    off = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(row_len, out=off[1:])
    adj = _CSRAdj(off, csr_cols, csr_ws, n_nodes)
    nodes_set = set(_node_arr.tolist())

    class LightweightGraph:
        def __init__(self, adj, off, cols, ws, n_nodes_total):
            self.adj = adj
            self.idx_to_node = idx_to_node
            self.node_to_idx = node_to_idx

            self.edge_weights = _EdgeWeightsView(ei, ej, ew)
            self._nodes = nodes_set
            self._node_list = list(nodes_set)

            self._off = off
            self._cols = cols
            self._ws = ws
            self._n_total = int(n_nodes_total)

        @property
        def n_nodes(self):
            return self._n_total

        def __len__(self):
            return len(self._nodes)

        def __contains__(self, node):
            return node in self.node_to_idx

        def __iter__(self):
            return iter(self._nodes)

        def neighbors(self, node):
            idx = self.node_to_idx[node]
            cs, ce = self._off[idx], self._off[idx + 1]
            return self._cols[cs:ce].tolist()

        def has_node(self, node):
            return node in self.node_to_idx

        def nodes(self):
            return self._nodes

        def edges(self, data=False):
            iu, iv, w = (self.edge_weights._iu, self.edge_weights._iv,
                         self.edge_weights._w)
            i2n = self.idx_to_node
            if data:
                for i, j, ww in zip(iu.tolist(), iv.tolist(), w.tolist()):
                    yield (i2n[i], i2n[j], {'weight': ww})
            else:
                for i, j in zip(iu.tolist(), iv.tolist()):
                    yield (i2n[i], i2n[j])

        def weighted_degrees(self):
            if self._ws.size:
                pre = np.empty(self._ws.size + 1, dtype=np.float64)
                pre[0] = 0.0
                np.cumsum(self._ws, out=pre[1:])
                return pre[self._off[1:]] - pre[self._off[:-1]]
            return np.zeros(self._n_total, dtype=np.float64)

        def degree(self, node=None, weight=None):
            if node is None:
                if weight is None:
                    row_len = np.diff(self._off)
                    return {n: int(row_len[self.node_to_idx[n]]) for n in self._nodes}
                wdeg = self.weighted_degrees()
                return {n: float(wdeg[self.node_to_idx[n]]) for n in self._nodes}
            idx = self.node_to_idx[node]
            if weight is None:
                return int(self._off[idx + 1] - self._off[idx])
            return float(np.sum(self._ws[self._off[idx]:self._off[idx + 1]]))

        def number_of_nodes(self):
            return len(self._nodes)

        def number_of_edges(self):
            return len(self.edge_weights)

        def is_directed(self):
            return False

        def is_multigraph(self):
            return False

        def order(self):
            return len(self._nodes)

        def size(self, weight=None):
            if weight is None:
                return len(self.edge_weights)
            return float(self.edge_weights._w.sum())

    return LightweightGraph(adj, off, csr_cols, csr_ws, n_nodes)

def merge_small_communities_cpm(node_to_comm, edge_df, tau, resolution_scale, n_nodes,
                                G=None):

    n_d = len(node_to_comm)
    _nodes_arr = np.fromiter(node_to_comm.keys(), dtype=np.int64, count=n_d)
    _lab_arr = np.fromiter(node_to_comm.values(), dtype=np.int64, count=n_d)
    uniq_lab, dense = np.unique(_lab_arr, return_inverse=True)
    dense = np.asarray(dense).ravel()
    K = int(uniq_lab.size)
    comm_dense = np.full(int(n_nodes), -1, dtype=np.int64)
    if _nodes_arr.size and (int(_nodes_arr.min()) < 0
                            or int(_nodes_arr.max()) >= int(n_nodes)):

        for _n, _d in zip(_nodes_arr.tolist(), dense.tolist()):
            comm_dense[_n] = _d
    else:
        comm_dense[_nodes_arr] = dense

    _sz_cnt = np.bincount(dense, minlength=K)
    csize = {int(uniq_lab[d]): int(_sz_cnt[d]) for d in range(K)}

    _od = np.argsort(dense, kind='stable')
    _sd = dense[_od]
    _sn = _nodes_arr[_od]
    _starts = np.searchsorted(_sd, np.arange(K), side='left')
    _starts = _starts[_sd[_starts] == np.arange(K)]
    _cmin_arr = np.minimum.reduceat(_sn, _starts)
    cmin = {int(uniq_lab[d]): int(_cmin_arr[d]) for d in range(K)}
    del _od, _sd, _sn, _starts, _cmin_arr

    _ew_v = getattr(G, 'edge_weights', None) if G is not None else None
    _use_csr = (_ew_v is not None and hasattr(_ew_v, '_iu') and len(_ew_v) > 0
                and int(getattr(G, 'n_nodes', n_nodes)) == int(n_nodes))
    if _use_csr:
        _pei = np.asarray(_ew_v._iu, dtype=np.int64)
        _pej = np.asarray(_ew_v._iv, dtype=np.int64)
        total_w = float(len(_ew_v))
    else:
        if isinstance(edge_df, pd.DataFrame):
            _eu = np.asarray(edge_df['u'].values, dtype=np.int64)
            _ev = np.asarray(edge_df['v'].values, dtype=np.int64)
        else:
            _eu = np.asarray(edge_df)[:, 0].astype(np.int64)
            _ev = np.asarray(edge_df)[:, 1].astype(np.int64)
        _lo = np.minimum(_eu, _ev)
        _hi = np.maximum(_eu, _ev)
        _ekey = _lo * np.int64(n_nodes) + _hi
        uk = np.unique(_ekey)
        del _ekey, _lo, _hi, _eu, _ev
        _pei = uk // n_nodes
        _pej = uk % n_nodes
        total_w = float(uk.size)
        del uk
    density = total_w / (n_nodes * (n_nodes - 1) / 2.0) if n_nodes > 1 else 0.0
    gamma = resolution_scale * math.sqrt(density) if density > 0 else 0.0

    _ci = comm_dense[_pei]
    _cj = comm_dense[_pej]
    _a, _b, _wsum = _comm_pair_weights(_pei, _pej,
                                       np.ones(_pei.size, dtype=np.float64),
                                       _ci, _cj, K)
    conn = defaultdict(dict)
    _a_l = uniq_lab[_a]
    _b_l = uniq_lab[_b]
    for _x, _y, _w in zip(_a_l.tolist(), _b_l.tolist(), _wsum.astype(np.int64).tolist()):
        conn[_x][_y] = _w
        conn[_y][_x] = _w
    del _pei, _pej, _ci, _cj, _a, _b, _wsum, _a_l, _b_l

    parent = {c: c for c in csize}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sizes = dict(csize)
    order = sorted(csize.keys(), key=lambda c: (csize[c], cmin[c], c))
    n_merged = 0
    for a in order:
        ra = find(a)
        if ra != a:
            continue
        while sizes[ra] < tau:

            agg = defaultdict(int)
            for b, w in conn[ra].items():
                rb = find(b)
                if rb != ra and sizes[rb] >= tau:
                    agg[rb] += w
            if not agg:
                break
            cand = sorted((-(w - gamma * sizes[ra] * sizes[rb]), rb)
                          for rb, w in agg.items())
            neg_gain, rb = cand[0]
            if -neg_gain <= 1e-12:
                break

            if sizes[ra] < sizes[rb] or (sizes[ra] == sizes[rb] and rb < ra):
                ra, rb = rb, ra
            w_ab = conn[ra].pop(rb, 0)
            conn[rb].pop(ra, None)
            parent[rb] = ra
            sizes[ra] += sizes[rb]
            for x, w in conn.pop(rb, {}).items():
                conn[ra][x] = conn[ra].get(x, 0) + w
                conn[x][ra] = conn[x].get(ra, 0) + w
                conn[x].pop(rb, None)
            del sizes[rb]
            n_merged += 1

    roots = sorted({find(c) for c in csize})
    remap = {r: i for i, r in enumerate(roots)}
    out = {n: remap[find(c)] for n, c in node_to_comm.items()}
    return out, n_merged

def _quotient_conn(G, comm_dict):
    vw = G.edge_weights
    uniq_lab, dense, comm_dense = _dense_comm_array(comm_dict, G.n_nodes)
    K = int(uniq_lab.size)
    ci = comm_dense[vw._iu]
    cj = comm_dense[vw._iv]
    lo, hi, wsum = _comm_pair_weights(vw._iu, vw._iv, vw._w, ci, cj, K)
    del comm_dense, ci, cj, dense
    return uniq_lab, K, lo, hi, wsum

def merge_communities_mutual_best(comm_dict, global_G):
    import math as _math
    import os as _os
    idx_to_node = global_G.idx_to_node
    edge_weights = global_G.edge_weights

    comm_sizes = defaultdict(int)
    for c in comm_dict.values():
        comm_sizes[c] += 1
    K = len(comm_sizes)
    if K < 2:
        return dict(comm_dict)

    if hasattr(global_G, 'edge_weights') and hasattr(edge_weights, '_iu') \
            and edge_weights._iu is not None and isinstance(idx_to_node, _IdentityIndex):

        vw = edge_weights
        uniq_lab, _dense, comm_dense = _dense_comm_array(comm_dict, global_G.n_nodes)
        Kv = int(uniq_lab.size)
        _ci = comm_dense[vw._iu]
        _cj = comm_dense[vw._iv]
        _pos = np.flatnonzero((_ci >= 0) & (_cj >= 0) & (_ci != _cj))

        _kn = (np.bincount(vw._iu, weights=vw._w.astype(np.float64),
                           minlength=global_G.n_nodes)
               + np.bincount(vw._iv, weights=vw._w.astype(np.float64),
                             minlength=global_G.n_nodes))
        _valid = comm_dense >= 0
        _kdense = np.bincount(comm_dense[_valid], weights=_kn[_valid],
                              minlength=Kv)

        _same = (_ci == _cj) & (_ci >= 0)
        _win = np.bincount(_ci[_same], weights=vw._w[_same], minlength=Kv)
        m_total = float(np.sum(vw._w))
        k_comm = dict(zip(uniq_lab.tolist(), _kdense.tolist()))
        w_in_comm = dict(zip(uniq_lab.tolist(), _win.tolist()))
        del _kn, _kdense, _valid, _same, _win
        if _pos.size:
            _lo = np.minimum(_ci[_pos], _cj[_pos]).astype(np.int64)
            _hi = np.maximum(_ci[_pos], _cj[_pos]).astype(np.int64)
            _key = _lo * np.int64(Kv) + _hi
            _uk, _first, _inv = np.unique(_key, return_index=True,
                                          return_inverse=True)
            _inv = np.asarray(_inv).ravel()
            _ws = np.bincount(_inv, weights=vw._w[_pos], minlength=_uk.size)
            _o = np.argsort(_first, kind='stable')
            _uk = _uk[_o]
            _ws = _ws[_o]
            _ll = uniq_lab[(_uk // Kv).astype(np.int64)]
            _lh = uniq_lab[(_uk % Kv).astype(np.int64)]
            conn = dict(zip(zip(_ll.tolist(), _lh.tolist()), _ws.tolist()))
            del _lo, _hi, _key, _uk, _inv, _ws, _o, _ll, _lh, _pos
        else:
            conn = {}
        del _ci, _cj, comm_dense, _dense
    else:
        node_to_comm = {}
        for n, c in comm_dict.items():
            node_to_comm[n] = c
        conn = defaultdict(float)
        node_deg = defaultdict(float)
        w_in_comm = defaultdict(float)
        m_total = 0.0
        for (i, j), w in edge_weights.items():
            u, v = idx_to_node[i], idx_to_node[j]
            node_deg[u] += w
            node_deg[v] += w
            m_total += w
            cu, cv = node_to_comm.get(u), node_to_comm.get(v)
            if cu is None or cv is None or cu == cv:
                if cu is not None and cu == cv:
                    w_in_comm[cu] += w
                continue
            if cu > cv:
                cu, cv = cv, cu
            conn[(cu, cv)] += w
        m_total = float(m_total)
        k_comm = defaultdict(float)
        for n, c in node_to_comm.items():
            k_comm[c] += node_deg[n]
        del node_deg

    m_q = len(conn)
    total_w = sum(conn.values())
    dens = total_w / (K * (K - 1) / 2) if K > 1 else 0.0

    w_share = total_w / m_total if m_total > 0 else 0.0
    guard_hit = w_share >= 0.5

    _mb_fb_thr = int(_math.ceil(_math.sqrt(2.0 * m_total))) if m_total > 0 else 0

    _N_tot = float(sum(comm_sizes.values()))
    _p_graph = (m_total / (_N_tot * (_N_tot - 1.0) / 2.0)) if _N_tot > 1.5 else 0.0

    def _sig_term(_n, _m_in):

        _slots = _n * (_n - 1.0) / 2.0
        if _slots <= 0.0 or _p_graph <= 0.0 or _p_graph >= 1.0:
            return 0.0
        _pc = min(_m_in / _slots, 1.0)
        if _pc >= 1.0:

            return _slots * (-_math.log(_p_graph))
        _t1 = _pc * _math.log(_pc / _p_graph) if _pc > 0.0 else 0.0
        _t2 = (1.0 - _pc) * _math.log((1.0 - _pc) / (1.0 - _p_graph))
        return _slots * (_t1 + _t2)

    print(f"  [MB-Merge] w_share={w_share:.3f} (guard ≥0.5), "
          f"K={K}, fb_thr=⌈√(2m)⌉={_mb_fb_thr}, "
          f"criterion=mutual-best + significance ΔS>0")

    _fb_dens_hit = (ENABLE_MB_FALLBACK and MB_FB_DENS_GUARD is not None
                    and dens >= MB_FB_DENS_GUARD and K < _mb_fb_thr)
    if guard_hit or m_q == 0 or _fb_dens_hit:

        if (ENABLE_MB_FALLBACK and m_q > 0 and K < _mb_fb_thr
                and (guard_hit or _fb_dens_hit)):
            _gdesc = (f"w_share={w_share:.3f} >= 0.5" if guard_hit
                      else f"dens={dens:.2f} >= {MB_FB_DENS_GUARD}")
            print(f"  [MB-Merge] {_gdesc} → "
                  f"fallback greedy dSqrt merge for small graph (K={K})")
            import math as _math

            if _os.environ.get('ASNCD_MB_SIG_FALLBACK'):
                pair_scores = []
                for (c1, c2), w in conn.items():
                    n1, n2 = comm_sizes[c1], comm_sizes[c2]
                    _mi1 = w_in_comm.get(c1, 0.0)
                    _mi2 = w_in_comm.get(c2, 0.0)
                    _dsg = (_sig_term(n1 + n2, _mi1 + _mi2 + w)
                            - _sig_term(n1, _mi1) - _sig_term(n2, _mi2))
                    if _dsg > 0.0:
                        pair_scores.append((_dsg, c1, c2))
                pair_scores.sort(reverse=True)
                max_fallback = len(pair_scores)
                _score_cut = -float('inf')
            else:
                pair_scores = []
                for (c1, c2), w in conn.items():
                    n1, n2 = comm_sizes[c1], comm_sizes[c2]
                    score = w / _math.sqrt(n1 * n2)
                    pair_scores.append((score, c1, c2))
                pair_scores.sort(reverse=True)
                max_fallback = max(2, int(K * MB_FB_MAX_FRAC))
                _score_cut = MB_FB_SCORE
            merged_into = {}
            n_merged = 0
            for score, c1, c2 in pair_scores[:max_fallback * 3]:
                if score < _score_cut:
                    break
                r1, r2 = c1, c2
                while r1 in merged_into:
                    r1 = merged_into[r1]
                while r2 in merged_into:
                    r2 = merged_into[r2]
                if r1 == r2:
                    continue
                merged_into[r2] = r1
                n_merged += 1
                if n_merged >= max_fallback:
                    break
            if n_merged > 0:
                groups = defaultdict(list)
                for n, c in comm_dict.items():
                    r = c
                    while r in merged_into:
                        r = merged_into[r]
                    groups[r].append(n)
                out = {}
                for new_id, nodes in enumerate(groups.values()):
                    for n in nodes:
                        out[n] = new_id
                print(f"  [MB-Merge fallback] merged {n_merged} pairs, "
                      f"{K} → {len(groups)} comms")
                return out
            else:
                print(f"  [MB-Merge] {_gdesc} → skip "
                      f"(no fallback candidates)")
                return dict(comm_dict)
        else:
            print(f"  [MB-Merge] guard hit (w_share={w_share:.3f}), "
                  f"K={K} >= fb_thr={_mb_fb_thr} → skip "
                  f"(partition scale mismatch, merging unsafe)")
            return dict(comm_dict)

    nbr = defaultdict(dict)
    for (c1, c2), w in conn.items():
        nbr[c1][c2] = w
        nbr[c2][c1] = w
    best = {}
    for c, d in nbr.items():
        best[c] = max(d.items(), key=lambda x: x[1])[0]

    cands = []

    def _mb_xlogx(x):

        return x * _math.log2(x) if x > 0.0 else 0.0

    _s2 = 2.0 * m_total
    _qbar = w_share

    _sig_cache = {}
    def _sig_of(_c):
        _v = _sig_cache.get(_c)
        if _v is None:
            _v = _sig_term(comm_sizes[_c], w_in_comm.get(_c, 0.0))
            _sig_cache[_c] = _v
        return _v
    for (c1, c2), w in conn.items():
        if best.get(c1) != c2 or best.get(c2) != c1:
            continue
        n1, n2 = comm_sizes[c1], comm_sizes[c2]
        _mi1 = w_in_comm.get(c1, 0.0)
        _mi2 = w_in_comm.get(c2, 0.0)
        d_sig = (_sig_term(n1 + n2, _mi1 + _mi2 + w)
                 - _sig_of(c1) - _sig_of(c2))
        if d_sig > 0.0:
            cands.append((d_sig, c1, c2))

        if _os.environ.get('ASNCD_MB_DIAG'):
            _sa, _sb = n1, n2
            _dsq = w / _math.sqrt(max(_sa * _sb, 1))
            _ka = k_comm.get(c1, 0.0)
            _kb = k_comm.get(c2, 0.0)
            _ra = 2.0 * m_total * w / (_ka * _kb) if _ka > 0 and _kb > 0 else 0.0
            _pab = w / _s2
            _qa = max((_ka - 2.0 * w_in_comm.get(c1, 0.0)) / _s2, 0.0)
            _qb = max((_kb - 2.0 * w_in_comm.get(c2, 0.0)) / _s2, 0.0)
            _Pa, _Pb = _ka / _s2, _kb / _s2
            _qab = _qa + _qb - 2.0 * _pab
            dL = (_mb_xlogx(_qbar - 2.0 * _pab) - _mb_xlogx(_qbar)
                  - 2.0 * (_mb_xlogx(_qab) - _mb_xlogx(_qa) - _mb_xlogx(_qb))
                  + (_mb_xlogx(_qab + _Pa + _Pb)
                     - _mb_xlogx(_qa + _Pa) - _mb_xlogx(_qb + _Pb)))
            _na = sorted(nbr[c1].values(), reverse=True)
            _nb = sorted(nbr[c2].values(), reverse=True)
            _sba = _na[1] if len(_na) > 1 else 0.0
            _sbb = _nb[1] if len(_nb) > 1 else 0.0
            _pass = d_sig > 0.0
            _tag = "PASS" if _pass else "REJ "
            print(f"  [MB-DIAG] {_tag} w={w:.0f} |A|={_sa} |B|={_sb} "
                  f"dS={d_sig:.4f} dSqrt={_dsq:.4f} R={_ra:.3f} dL={dL:.5f} "
                  f"2ndA={_sba:.0f} 2ndB={_sbb:.0f} "
                  f"domA={w > _sba} domB={w > _sbb} "
                  f"c=({c1},{c2})")
    if not cands:
        print("  [MB-Merge] 0 candidate pairs, nothing to merge")
        return dict(comm_dict)

    cands.sort(reverse=True)
    merged_into = {}
    n_merged = 0
    for _, c1, c2 in cands:
        r1, r2 = c1, c2
        while r1 in merged_into:
            r1 = merged_into[r1]
        while r2 in merged_into:
            r2 = merged_into[r2]
        if r1 == r2:
            continue
        merged_into[r2] = r1
        n_merged += 1

    groups = defaultdict(list)
    for n, c in comm_dict.items():
        r = c
        while r in merged_into:
            r = merged_into[r]
        groups[r].append(n)
    out = {}
    for new_id, nodes in enumerate(groups.values()):
        for n in nodes:
            out[n] = new_id
    print(f"  [MB-Merge] merged {n_merged} pairs "
          f"(dens={dens:.2e}, {K} → {len(groups)} comms)")
    return out

SGM_ONLY_LARGEST = True
SGM_DIAG = False

SGM_GATE = 'struct'

SGM_COH_MAX = 0.15

SGM_FRAG_T_OVERRIDE = None

HYBRID_GIANTS_K = 0
HYBRID_AUTO = False

HYBRID_MIN_FRAG = 1000

HYBRID_FRAC_MIN = None

_HYBRID_OVERLAP_PRIOR = {'co-purchase', 'collaboration'}
_HYBRID_AUTO_K = 64

def rebuild_giant_communities(comm_dict, global_G, total_nodes, total_edges=None):
    idx_to_node = global_G.idx_to_node
    edge_weights = global_G.edge_weights

    comm_to_nodes = defaultdict(list)
    for n, c in comm_dict.items():
        comm_to_nodes[c].append(n)
    K = len(comm_to_nodes)
    if K < 2:
        return dict(comm_dict), None

    if (isinstance(idx_to_node, _IdentityIndex)
            and hasattr(edge_weights, '_iu') and len(edge_weights)):
        uniq_lab, Kd, lo, hi, wsum = _quotient_conn(global_G, comm_dict)
        conn = defaultdict(float)
        for _x, _y, _w in zip(uniq_lab[lo].tolist(), uniq_lab[hi].tolist(),
                              wsum.tolist()):
            conn[(_x, _y)] = _w
        del uniq_lab, Kd, lo, hi, wsum
    else:
        node_to_comm = {}
        for n, c in comm_dict.items():
            node_to_comm[n] = c
        conn = defaultdict(float)
        for (i, j), w in edge_weights.items():
            u, v = idx_to_node[i], idx_to_node[j]
            cu, cv = node_to_comm.get(u), node_to_comm.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            if cu > cv:
                cu, cv = cv, cu
            conn[(cu, cv)] += w
    if not conn:
        return dict(comm_dict), None

    comm_ids = sorted(comm_to_nodes.keys())
    g = ig.Graph()
    g.add_vertices(K)
    cid_pos = {c: p for p, c in enumerate(comm_ids)}
    q_edges = sorted(conn.keys())
    g.add_edges([(cid_pos[a], cid_pos[b]) for (a, b) in q_edges])
    g.es['weight'] = [conn[(a, b)] for (a, b) in q_edges]

    part = la.find_partition(g, la.ModularityVertexPartition, weights='weight',
                             n_iterations=10, seed=42)
    macros = defaultdict(list)
    for p, mc in enumerate(part.membership):
        macros[mc].append(comm_ids[p])

    if SGM_ONLY_LARGEST:
        biggest_mc = max(macros,
                         key=lambda k: sum(len(comm_to_nodes[c]) for c in macros[k]))
    else:
        biggest_mc = None

    if SGM_DIAG and isinstance(idx_to_node, _IdentityIndex) \
            and hasattr(edge_weights, '_iu') and len(edge_weights):
        _du = np.asarray(edge_weights._iu, dtype=np.int64)
        _dv = np.asarray(edge_weights._iv, dtype=np.int64)
        _cand = sorted(macros.items(), key=lambda kv: -len(kv[1]))[:3]
        if biggest_mc is not None and all(mc != biggest_mc for mc, _ in _cand):
            _cand.append((biggest_mc, macros[biggest_mc]))
        for _mc, _mem in _cand:
            _nodes = np.fromiter((n for c in _mem for n in comm_to_nodes[c]),
                                 dtype=np.int64)
            if _nodes.size == 0:
                continue
            _mask = np.zeros(total_nodes, dtype=bool)
            _mask[_nodes] = True
            _inc = _mask[_du] | _mask[_dv]
            _ext = _mask[_du] ^ _mask[_dv]
            _ninc = int(_inc.sum())
            _theta = int(_ext[_inc].sum()) / max(_ninc, 1)
            _S = int(_nodes.size)
            _tag = ' BIGGEST' if _mc == biggest_mc else ''
            print(f"  [SGMdiag] frags={len(_mem):6d} |S|={_S:8d} "
                  f"frac={100.0*_S/max(total_nodes,1):5.1f}% "
                  f"mean_frag={_S/max(len(_mem),1):7.1f} ext_frac={_theta:5.3f}{_tag}")

    frag_t = max(1, int(math.ceil(math.sqrt(2.0 * total_edges))))
    if SGM_FRAG_T_OVERRIDE is not None:
        frag_t = max(1, int(SGM_FRAG_T_OVERRIDE))

    _use_struct = (SGM_GATE == 'struct'
                   and isinstance(idx_to_node, _IdentityIndex)
                   and hasattr(edge_weights, '_iu') and len(edge_weights))
    if _use_struct:
        _su = np.asarray(edge_weights._iu, dtype=np.int64)
        _sv = np.asarray(edge_weights._iv, dtype=np.int64)

    _macro_stats = sorted(
        ((mc, len(members), sum(len(comm_to_nodes[c]) for c in members))
         for mc, members in macros.items()),
        key=lambda t: -t[2])
    print(f"  [SGM-VERIFY] enter: N={total_nodes}, E={total_edges}, K={K}, "
          f"macros={len(macros)}, gate={SGM_GATE}, use_struct={_use_struct}, "
          f"only_largest={SGM_ONLY_LARGEST}, coh_max={SGM_COH_MAX}, "
          f"frag_t={frag_t}")
    print(f"  [SGM-VERIFY] macro stats (mc, n_frags, n_nodes) top5: "
          f"{[(mc, nf, nn) for mc, nf, nn in _macro_stats[:5]]}"
          f"{' ...' if len(_macro_stats) > 5 else ''}")
    if biggest_mc is not None:
        _bm_nf = len(macros[biggest_mc])
        _bm_nn = sum(len(comm_to_nodes[c]) for c in macros[biggest_mc])
        print(f"  [SGM-VERIFY] biggest_mc={biggest_mc} "
              f"(frags={_bm_nf}, nodes={_bm_nn}, "
              f"frags_ok={_bm_nf >= frag_t})")
    else:
        print(f"  [SGM-VERIFY] biggest_mc=None (SGM_ONLY_LARGEST=False, "
              f"所有合格宏都参与)")

    if _use_struct:
        if len(_macro_stats) >= 2:
            for _q7_rank, (_q7_mc, _q7_nf, _q7_nn) in enumerate(
                    _macro_stats[:2], 1):
                _q7_nodes = []
                for _c in macros[_q7_mc]:
                    _q7_nodes.extend(comm_to_nodes[_c])
                _q7_mask = np.zeros(total_nodes, dtype=bool)
                _q7_mask[np.asarray(_q7_nodes, dtype=np.int64)] = True
                _q7_inc = _q7_mask[_su] | _q7_mask[_sv]
                _q7_ninc = int(_q7_inc.sum())
                _q7_theta = (
                    int((_q7_mask[_su] ^ _q7_mask[_sv])[_q7_inc].sum())
                    / max(_q7_ninc, 1)) if _q7_ninc else 0.0
                _q7_s1 = _q7_nf >= frag_t
                _q7_s2 = _q7_theta <= SGM_COH_MAX
                _q7_tag = ('BIGGEST' if (biggest_mc is not None
                                         and _q7_mc == biggest_mc)
                           else ('2nd' if _q7_rank == 2 else 'top'))
                print(f"  [SGM-Q7] rank={_q7_rank} ({_q7_tag}) mc={_q7_mc}: "
                      f"frags={_q7_nf}/{frag_t} "
                      f"S1={'PASS' if _q7_s1 else 'FAIL'} | "
                      f"nodes={_q7_nn} frac={100.0*_q7_nn/max(total_nodes,1):.2f}% "
                      f"theta={_q7_theta:.4f}/{SGM_COH_MAX} "
                      f"S2={'PASS' if _q7_s2 else 'FAIL'} | "
                      f"S1+S2={'PASS' if (_q7_s1 and _q7_s2) else 'FAIL'}")
        else:
            print(f"  [SGM-Q7] 第二大宏不存在（macros={len(macros)}）")
    else:
        print(f"  [SGM-Q7] use_struct=False，内聚统计不可计算（跳过审计）")

    out = {}
    next_id = 0
    n_giant = 0
    n_reject = 0
    giant_nodes_out = None

    _verify_cands = []

    for mc, members in macros.items():
        nodes = []
        for c in members:
            nodes.extend(comm_to_nodes[c])
        frac = len(nodes) / max(total_nodes, 1)
        eligible = (biggest_mc is None) or (mc == biggest_mc)
        _frags_ok = (len(members) >= frag_t)
        gate_ok = False
        _theta = None

        if eligible and _frags_ok:
            if _use_struct:
                _mask = np.zeros(total_nodes, dtype=bool)
                _mask[np.asarray(nodes, dtype=np.int64)] = True
                _inc = _mask[_su] | _mask[_sv]
                _ninc = int(_inc.sum())
                _theta = (int((_mask[_su] ^ _mask[_sv])[_inc].sum()) / max(_ninc, 1)
                          ) if _ninc else 0.0
                gate_ok = _theta <= SGM_COH_MAX
                _margin = SGM_COH_MAX - _theta
                _tag = ' BIGGEST' if (biggest_mc is not None
                                      and mc == biggest_mc) else ''
                print(f"  [SGM-VERIFY] cand mc={mc}{_tag}: "
                      f"frags={len(members)} nodes={len(nodes)} "
                      f"frac={100.0*frac:.2f}% theta={_theta:.4f} "
                      f"gate_max={SGM_COH_MAX:.4f} margin={_margin:+.4f} "
                      f"pass={gate_ok}")
                _verify_cands.append((mc, _theta, gate_ok))
            else:

                print(f"  [SGM-VERIFY] cand mc={mc}: frags={len(members)} "
                      f"nodes={len(nodes)} frac={100.0*frac:.2f}% "
                      f"gate_use_struct=False → gate_ok=False"
                      f"（census 分支已删除，SGM_COH_MAX 不参与判定）")
        elif eligible and not _frags_ok:

            print(f"  [SGM-VERIFY] biggest candidate mc={mc}: "
                  f"frags={len(members)} < frag_t={frag_t} "
                  f"→ 不进入门控判定（SGM_COH_MAX 不参与）")
        elif (not eligible) and _frags_ok:

            print(f"  [SGM-VERIFY] skip mc={mc} (非最大宏): "
                  f"frags={len(members)} nodes={len(nodes)} "
                  f"frac={100.0*frac:.2f}%（_theta 未计算，原行为）")

        if gate_ok:
            if _use_struct:
                print(f"  [SGM] macro coherence: frags={len(members)} "
                      f"ext_frac={_theta:.3f} (≤{SGM_COH_MAX:.2f}?{gate_ok})")

        if eligible and _frags_ok and gate_ok:
            for n in nodes:
                out[n] = next_id
            giant_nodes_out = list(nodes)
            next_id += 1
            n_giant += 1
            print(f"  [SGM] giant rebuilt: {len(members)} fragments → "
                  f"{len(nodes)} nodes ({frac:.1%} of graph)")
        else:
            if _frags_ok and gate_ok:
                n_reject += 1
                why = '非最大宏' if not eligible else '超出 frac 上限'
                print(f"  [SGM] REJECT: {len(members)} fragments → "
                      f"{len(nodes)} nodes ({frac:.1%} of graph)  [{why}]")
            for c in members:
                for n in comm_to_nodes[c]:
                    out[n] = next_id
                next_id += 1

    if _verify_cands:
        _thetas = [round(t, 4) for _, t, _ in _verify_cands]
        _n_pass = sum(1 for _, _, p in _verify_cands if p)
        print(f"  [SGM-VERIFY] summary: candidates={len(_verify_cands)}, "
              f"passed={_n_pass}, thetas={_thetas}, coh_max={SGM_COH_MAX}")
    else:
        print(f"  [SGM-VERIFY] summary: 无候选进入 struct 门控"
              f"（use_struct={_use_struct}, frag_t={frag_t}, "
              f"biggest_mc={biggest_mc}）")

    print(f"  [SGM] giants={n_giant}, rejected={n_reject}, {K} → {next_id} comms "
          f"(largest macro = {max(len(m) for m in macros.values())} fragments, "
          f"frag_threshold={frag_t} "
          f"[{'struct' if _use_struct else SGM_T_MODE}], N={total_nodes})")
    return out, giant_nodes_out

SGM_T2 = 1000

def split_glued_communities(comm_dict, edge_df, total_nodes, resolution_scale,
                            cut_frac_abs=0.3,
                            verbose=True):
    comm_to_nodes = defaultdict(list)
    for n, c in comm_dict.items():
        comm_to_nodes[c].append(n)

    if isinstance(edge_df, pd.DataFrame):
        _sgu = np.asarray(edge_df['u'].values, dtype=np.int64)
        _sgv = np.asarray(edge_df['v'].values, dtype=np.int64)
        _E = int(len(edge_df))
    else:
        _arr = np.asarray(edge_df)
        _sgu = _arr[:, 0].astype(np.int64)
        _sgv = _arr[:, 1].astype(np.int64)
        _E = int(_arr.shape[0])
    _nsg = max(int(total_nodes), 1)
    _E2 = _sgu.size * 2
    _sg_rows = np.empty(_E2, dtype=np.int64)
    _sg_rows[0::2] = _sgu
    _sg_rows[1::2] = _sgv
    _sg_cols = np.empty(_E2, dtype=np.int64)
    _sg_cols[0::2] = _sgv
    _sg_cols[1::2] = _sgu
    del _sgu, _sgv
    _sg_order = np.argsort(_sg_rows, kind='stable')
    _sg_cols = _sg_cols[_sg_order]
    del _sg_order
    _sg_len = np.bincount(_sg_rows, minlength=_nsg).astype(np.int64)
    del _sg_rows
    _sg_off = np.zeros(_nsg + 1, dtype=np.int64)
    np.cumsum(_sg_len, out=_sg_off[1:])

    def _adj_rows(m_arr):
        base = _sg_off[m_arr]
        counts = _sg_len[m_arr]
        total = int(counts.sum())
        if total == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        rep_base = np.repeat(base, counts)
        cum = np.cumsum(counts)
        within = np.arange(total) - np.repeat(cum - counts, counts)
        return (np.repeat(m_arr, counts), _sg_cols[rep_base + within])

    _avg_deg = (2.0 * _E) / _nsg
    size_thr = int(math.ceil(math.sqrt(_nsg * _avg_deg)))

    cut_thr = float(cut_frac_abs)

    if verbose:
        print(f"  [SGSplit] size_thr=ceil(sqrt(2E))={size_thr} "
              f"(N={_nsg}, E={_E}, avg_deg={_avg_deg:.2f}), "
              f"cut_thr={cut_thr:.3f}")

    out = dict(comm_dict)
    n_split = 0
    next_id = max(comm_to_nodes.keys()) + 1 if comm_to_nodes else 0

    for cid in sorted(comm_to_nodes.keys()):
        members = comm_to_nodes[cid]
        if len(members) < size_thr:
            continue
        nl = np.fromiter(members, dtype=np.int64, count=len(members))
        nl.sort()
        in_mask = np.zeros(_nsg, dtype=bool)
        in_mask[nl] = True

        _src, _nbrs = _adj_rows(nl)
        deg_sum = int(_nbrs.size)
        m_in = int(np.count_nonzero(in_mask[_nbrs])) // 2
        del _src, _nbrs
        m_out = deg_sum - 2 * m_in
        tot = m_in + m_out
        cut_frac = (m_out / tot) if tot else 1.0
        if m_in == 0 or cut_frac < cut_thr:
            continue

        _src2, _nbrs2 = _adj_rows(nl)
        _keep_es = (_nbrs2 < _src2) & in_mask[_nbrs2]
        _ev_outer = _src2[_keep_es]
        _ev_inner = _nbrs2[_keep_es]
        del _src2, _nbrs2, _keep_es
        loc_outer = np.searchsorted(nl, _ev_outer)
        loc_inner = np.searchsorted(nl, _ev_inner)
        es = list(zip(loc_outer.tolist(), loc_inner.tolist()))
        n_sub = int(nl.size)
        m_sub = len(es)
        if m_sub == 0:
            continue
        w_density = m_sub / (n_sub * (n_sub - 1) / 2.0)
        gamma = resolution_scale * math.sqrt(w_density)
        gamma = max(0.001, min(gamma, 20.0))
        try:
            g = ig.Graph(n=n_sub, edges=es, directed=False)
            part = la.find_partition(g, la.CPMVertexPartition,
                                     resolution_parameter=gamma,
                                     n_iterations=10, seed=RANDOM_SEED)
        except Exception:
            continue
        memb = defaultdict(list)
        for i, mc in enumerate(part.membership):
            memb[mc].append(nl[i])
        parts = [p for p in memb.values() if p]
        if len(parts) < 2:
            continue

        for n in members:
            del out[n]
        for p in parts:
            for n in p:
                out[n] = next_id
            next_id += 1
        n_split += 1
        if verbose:
            print(f"  [SGSplit] split {len(members)}-node comm "
                  f"(cut_frac={cut_frac:.2f}) → {len(parts)} parts "
                  f"(γ_sub={gamma:.4f})")

    if verbose:
        print(f"  [SGSplit] structural: size_thr={size_thr}, "
              f"cut_thr={cut_thr:.3f}: split {n_split} comms, "
              f"{len(comm_to_nodes)} → {len(set(out.values()))} comms")
    return out, n_split

def convert_comm_dict_to_comms_list(comm_dict):
    comm_to_nodes = defaultdict(list)
    for node, comm_id in comm_dict.items():
        comm_to_nodes[comm_id].append(node)

    return [nodes for nodes in comm_to_nodes.values()]

def describe_partition(comm_dict, tag, gt_comms=None):
    from collections import Counter
    if not comm_dict:
        print(f"  [PROF {tag}] empty")
        return
    sizes = sorted(Counter(comm_dict.values()).values())
    n_comms = len(sizes)
    tot = sum(sizes)
    le2 = 100.0 * sum(1 for s in sizes if s <= 2) / n_comms
    le5 = 100.0 * sum(1 for s in sizes if s <= 5) / n_comms
    line = (f"  [PROF {tag}] comms={n_comms}, nodes={tot}, avg={tot / n_comms:.2f}, "
            f"med={sizes[n_comms // 2]}, max={sizes[-1]}, "
            f"sz<=2:{le2:.1f}%, sz<=5:{le5:.1f}%")
    if gt_comms:
        gs = sorted(len(c) for c in gt_comms)
        gtot = sum(gs)
        g_le2 = 100.0 * sum(1 for s in gs if s <= 2) / len(gs)
        g_le5 = 100.0 * sum(1 for s in gs if s <= 5) / len(gs)
        line += (f"   ||GT comms={len(gs)}, avg={gtot / len(gs):.2f}, med={gs[len(gs) // 2]}, "
                 f"max={gs[-1]}, sz<=2:{g_le2:.1f}%, sz<=5:{g_le5:.1f}%")
    print(line)

def build_modularity_graph(edge_df, num_nodes):
    u = edge_df['u'].values.tolist()
    v = edge_df['v'].values.tolist()
    g = ig.Graph(n=int(num_nodes), edges=list(zip(u, v)), directed=False)
    g.simplify()
    return g

def comm_dict_to_membership(comm_dict, num_nodes):
    remap = {}
    mem = [0] * int(num_nodes)
    for v in range(int(num_nodes)):
        c = comm_dict.get(v)
        if c is None:
            mem[v] = -1
        else:
            idx = remap.get(c)
            if idx is None:
                idx = len(remap)
                remap[c] = idx
            mem[v] = idx
    nxt = len(remap)
    for v in range(int(num_nodes)):
        if mem[v] < 0:
            mem[v] = nxt
            nxt += 1
    return mem

try:
    from numba import njit, prange
    import random as _random_module

    @njit(cache=True)
    def _lpa_numba_run_from_init_multi(off, cols, init_labels, num_nodes, seed, max_outer_iter=5, max_inner_iter=20):
        np.random.seed(seed)

        labels = init_labels.copy()
        order = np.arange(num_nodes, dtype=np.int64)

        label_counts = np.zeros(num_nodes, dtype=np.int64)
        unique_labels = np.zeros(num_nodes, dtype=np.int64)

        prev_mod = -1.0

        for outer_iter in range(max_outer_iter):

            for inner_iter in range(max_inner_iter):
                np.random.shuffle(order)
                changed = False

                for idx in range(len(order)):
                    node = order[idx]
                    start, end = off[node], off[node + 1]
                    if start == end:
                        continue

                    n_unique = 0
                    for j in range(start, end):
                        lbl = labels[cols[j]]
                        if label_counts[lbl] == 0:
                            unique_labels[n_unique] = lbl
                            n_unique += 1
                        label_counts[lbl] += 1

                    max_label = -1
                    max_count = 0
                    for k in range(n_unique):
                        lbl = unique_labels[k]
                        count = label_counts[lbl]
                        if count > max_count or (count == max_count and lbl > max_label):
                            max_count = count
                            max_label = lbl

                    for k in range(n_unique):
                        label_counts[unique_labels[k]] = 0

                    if labels[node] != max_label:
                        labels[node] = max_label
                        changed = True

                if not changed:
                    break

            n_comms = 0
            seen = np.zeros(num_nodes, dtype=np.int64)
            for i in range(num_nodes):
                lbl = labels[i]
                if seen[lbl] == 0:
                    seen[lbl] = 1
                    n_comms += 1

            if outer_iter > 0 and abs(n_comms - prev_mod) < 10:
                break
            prev_mod = n_comms

        return labels

    @njit(cache=True)
    def _lpa_numba_run_from_init(off, cols, init_labels, num_nodes, seed, max_iter=100):

        np.random.seed(seed)

        labels = init_labels.copy()
        order = np.arange(num_nodes, dtype=np.int64)

        label_counts = np.zeros(num_nodes, dtype=np.int64)
        unique_labels = np.zeros(num_nodes, dtype=np.int64)

        for iteration in range(max_iter):

            np.random.shuffle(order)

            changed = False

            for idx in range(len(order)):
                node = order[idx]
                start, end = off[node], off[node + 1]
                if start == end:
                    continue

                n_neighbors = end - start
                if n_neighbors == 0:
                    continue

                n_unique = 0
                for j in range(start, end):
                    lbl = labels[cols[j]]
                    if label_counts[lbl] == 0:
                        unique_labels[n_unique] = lbl
                        n_unique += 1
                    label_counts[lbl] += 1

                max_label = -1
                max_count = 0
                for k in range(n_unique):
                    lbl = unique_labels[k]
                    count = label_counts[lbl]
                    if count > max_count or (count == max_count and lbl > max_label):
                        max_count = count
                        max_label = lbl

                for k in range(n_unique):
                    label_counts[unique_labels[k]] = 0

                if labels[node] != max_label:
                    labels[node] = max_label
                    changed = True

            if not changed:
                break

        return labels

    @njit(cache=True)
    def _lpa_numba_run(off, cols, num_nodes, seed, max_iter=100):

        np.random.seed(seed)

        labels = np.arange(num_nodes, dtype=np.int64)
        order = np.arange(num_nodes, dtype=np.int64)

        label_counts = np.zeros(num_nodes, dtype=np.int64)
        unique_labels = np.zeros(num_nodes, dtype=np.int64)

        for iteration in range(max_iter):

            np.random.shuffle(order)

            changed = False

            for idx in range(len(order)):
                node = order[idx]
                start, end = off[node], off[node + 1]
                if start == end:
                    continue

                n_neighbors = end - start
                if n_neighbors == 0:
                    continue

                n_unique = 0
                for j in range(start, end):
                    lbl = labels[cols[j]]
                    if label_counts[lbl] == 0:
                        unique_labels[n_unique] = lbl
                        n_unique += 1
                    label_counts[lbl] += 1

                max_label = -1
                max_count = 0
                for k in range(n_unique):
                    lbl = unique_labels[k]
                    count = label_counts[lbl]
                    if count > max_count or (count == max_count and lbl > max_label):
                        max_count = count
                        max_label = lbl

                for k in range(n_unique):
                    label_counts[unique_labels[k]] = 0

                if labels[node] != max_label:
                    labels[node] = max_label
                    changed = True

            if not changed:
                break

        return labels

    NUMBA_LPA_AVAILABLE = True
except ImportError:
    NUMBA_LPA_AVAILABLE = False

def _lpa_numpy_version(off, cols, num_nodes, seed, max_iter=100):
    import random as _random
    _random.seed(seed)

    labels = np.arange(num_nodes, dtype=np.int64)

    order = np.arange(num_nodes, dtype=np.int64)

    for _ in range(max_iter):
        changed = False
        _random.shuffle(order)

        for node in order:

            start, end = off[node], off[node + 1]
            if start == end:
                continue

            neighbors = cols[start:end]

            neighbor_labels = labels[neighbors]
            if len(neighbor_labels) == 0:
                continue

            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()

            candidates = unique_labels[counts == max_count]
            if len(candidates) == 1:
                new_label = candidates[0]
            else:
                new_label = _random.choice(candidates.tolist())

            if labels[node] != new_label:
                labels[node] = new_label
                changed = True

        if not changed:
            break

    return labels

try:
    from numba import njit

    @njit(cache=True)
    def _compute_modularity_numba(off, cols, ws, labels, num_nodes):

        degrees = np.zeros(num_nodes, dtype=np.float64)
        for i in range(num_nodes):
            degrees[i] = off[i + 1] - off[i]

        total_edges = off[num_nodes] / 2.0

        if total_edges == 0:
            return 0.0

        Q = 0.0
        for i in range(num_nodes):
            start_i = off[i]
            end_i = off[i + 1]
            for idx_j in range(start_i, end_i):
                j = cols[idx_j]
                if labels[i] == labels[j]:

                    expected = degrees[i] * degrees[j] / (2.0 * total_edges)
                    Q += 1.0 - expected

        Q /= (2.0 * total_edges)
        return Q

    NUMBA_MODULARITY_AVAILABLE = True
except ImportError:
    NUMBA_MODULARITY_AVAILABLE = False

def _compute_modularity_numpy(off, cols, ws, labels, num_nodes):
    if NUMBA_MODULARITY_AVAILABLE:
        return _compute_modularity_numba(off, cols, ws, labels, num_nodes)

    degrees = np.diff(off).astype(np.float64)
    total_edges = off[-1] / 2.0

    if total_edges == 0:
        return 0.0

    Q = 0.0
    for i in range(num_nodes):
        start_i, end_i = off[i], off[i + 1]
        for idx_j in range(start_i, end_i):
            j = cols[idx_j]
            if labels[i] == labels[j]:

                expected = degrees[i] * degrees[j] / (2.0 * total_edges)
                Q += 1.0 - expected

    Q /= (2.0 * total_edges)
    return Q

def _lpa_single_run_csr(args):
    off, cols, ws, num_nodes, seed, collapse_frac, i = args
    from collections import Counter as _Counter

    try:

        if NUMBA_LPA_AVAILABLE:
            mem = _lpa_numba_run(off, cols, num_nodes, seed)
        else:
            mem = _lpa_numpy_version(off, cols, num_nodes, seed)

        sizes = _Counter(mem.tolist())
        mx = max(sizes.values())
        cap = collapse_frac * num_nodes
        if mx > cap:
            return (seed, len(sizes), mx, float('nan'), 'collapsed', None)
        mod = _compute_modularity_numpy(off, cols, ws, mem, num_nodes)
        return (seed, len(sizes), mx, mod, 'ok', mem.tolist())
    except Exception as e:
        return (seed, 0, 0, -1.0, 'failed', str(e))

def _lpa_single_run_csr_from_init(args):
    off, cols, ws, init_labels, num_nodes, seed, collapse_frac, i = args
    from collections import Counter as _Counter

    try:

        if NUMBA_LPA_AVAILABLE:
            mem = _lpa_numba_run_from_init_multi(
                off, cols, init_labels, num_nodes, seed,
                max_outer_iter=5,
                max_inner_iter=20)
        else:

            mem = _lpa_numba_run(off, cols, num_nodes, seed)

        sizes = _Counter(mem.tolist())
        mx = max(sizes.values())
        cap = collapse_frac * num_nodes
        if mx > cap:
            return (seed, len(sizes), mx, float('nan'), 'collapsed', None)
        mod = _compute_modularity_numpy(off, cols, ws, mem, num_nodes)
        return (seed, len(sizes), mx, mod, 'ok', mem.tolist())
    except Exception as e:
        return (seed, 0, 0, -1.0, 'failed', str(e))

def _lpa_single_run(args):
    edges_np, num_nodes, seed, collapse_frac, i = args
    import random as _random
    from collections import Counter as _Counter
    import igraph as ig

    _random.seed(seed)
    try:

        g = ig.Graph(n=num_nodes, edges=edges_np.tolist(), directed=False)
        mem = list(g.community_label_propagation().membership)
        sizes = _Counter(mem)
        mx = max(sizes.values())
        cap = collapse_frac * num_nodes
        if mx > cap:
            return (seed, len(sizes), mx, float('nan'), 'collapsed', None)
        mod = float(g.modularity(mem))
        return (seed, len(sizes), mx, mod, 'ok', mem)
    except Exception as e:
        return (seed, 0, 0, -1.0, 'failed', str(e))

def maxmod_lpa_candidate_parallel(edges_np, num_nodes, n_runs=None, seed_base=None,
                                   collapse_frac=None, verbose=True, n_workers=None):
    from concurrent.futures import ProcessPoolExecutor
    import os

    n_runs = int(n_runs if n_runs is not None else LPA_CAND_RUNS)
    seed_base = int(seed_base if seed_base is not None else LPA_CAND_SEED_BASE)
    collapse_frac = float(collapse_frac) if collapse_frac is not None else 1.0

    if n_workers is None:
        n_workers = min(n_runs, os.cpu_count() or 4)

    tasks = [(edges_np, num_nodes, seed_base + i, collapse_frac, i) for i in range(n_runs)]

    if verbose:
        print(f"    [LPA-parallel] running {n_runs} LPA runs with {n_workers} workers")

    best_mem, best_mod, best_k, best_max = None, -1.0, 0, 0
    runs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_lpa_single_run, task) for task in tasks]
        for i, future in enumerate(futures):
            result = future.result()
            seed, k, mx, mod, status, mem_or_err = result

            if status == 'failed':
                if verbose:
                    print(f"    [LPA-cand] run{i} (seed={seed}) failed: {mem_or_err}")
                runs.append((seed, 0, 0, -1.0, 'failed'))
                continue

            if status == 'collapsed':
                runs.append((seed, k, mx, float('nan'), 'collapsed'))
                if verbose:
                    print(f"    [LPA-cand] run{i} seed={seed}: K={k:6d} "
                          f"max={mx:6d} -> COLLAPSED (>{collapse_frac:.0%}N), skipped")
                continue

            runs.append((seed, k, mx, mod, 'ok'))
            if verbose:
                print(f"    [LPA-cand] run{i} seed={seed}: K={k:6d} "
                      f"max={mx:6d} mod={mod:.4f}")
            if mod > best_mod:
                best_mod, best_mem, best_k, best_max = mod, mem_or_err, k, mx

    info = dict(n_runs=n_runs, n_ok=sum(1 for r in runs if r[4] == 'ok'),
                best_mod=best_mod if best_mem is not None else None,
                best_k=best_k, best_max=best_max, runs=runs,
                collapse_frac=collapse_frac)
    return best_mem, info

def maxmod_lpa_candidate(g, num_nodes, n_runs=None, seed_base=None,
                         collapse_frac=None, verbose=True):
    import random as _random
    from collections import Counter as _Counter

    n_runs = int(n_runs if n_runs is not None else LPA_CAND_RUNS)
    seed_base = int(seed_base if seed_base is not None else LPA_CAND_SEED_BASE)
    collapse_frac = float(collapse_frac) if collapse_frac is not None else 1.0
    cap = collapse_frac * int(num_nodes)

    best_mem, best_mod, best_k, best_max = None, -1.0, 0, 0
    runs = []
    for i in range(n_runs):
        _random.seed(seed_base + i)
        try:
            mem = list(g.community_label_propagation().membership)
        except Exception as e:
            if verbose:
                print(f"    [LPA-cand] run{i} (seed={seed_base + i}) failed: {e}")
            continue
        sizes = _Counter(mem)
        mx = max(sizes.values())
        if mx > cap:
            runs.append((seed_base + i, len(sizes), mx, float('nan'), 'collapsed'))
            if verbose:
                print(f"    [LPA-cand] run{i} seed={seed_base + i}: K={len(sizes):6d} "
                      f"max={mx:6d} -> COLLAPSED (>{collapse_frac:.0%}N), skipped")
            continue
        mod = float(g.modularity(mem))
        runs.append((seed_base + i, len(sizes), mx, mod, 'ok'))
        if verbose:
            print(f"    [LPA-cand] run{i} seed={seed_base + i}: K={len(sizes):6d} "
                  f"max={mx:6d} mod={mod:.4f}")
        if mod > best_mod:
            best_mod, best_mem, best_k, best_max = mod, mem, len(sizes), mx

    info = dict(n_runs=n_runs, n_ok=sum(1 for r in runs if r[4] == 'ok'),
                best_mod=best_mod if best_mem is not None else None,
                best_k=best_k, best_max=best_max, runs=runs,
                collapse_frac=collapse_frac)
    return best_mem, info

def _dense_labels_from_comm_dict(comm_dict, num_nodes):
    num_nodes = int(num_nodes)
    remap = {}
    mem = np.empty(num_nodes, dtype=np.int64)
    nxt = 0
    for v in range(num_nodes):
        c = comm_dict.get(v)
        if c is None:
            mem[v] = -1
            continue
        idx = remap.get(c)
        if idx is None:
            idx = nxt
            remap[c] = idx
            nxt += 1
        mem[v] = idx
    for v in range(num_nodes):
        if mem[v] < 0:
            mem[v] = nxt
            nxt += 1
    return mem

ORPHAN_ENABLE = True
ORPHAN_MAX_SIZE = 2
ORPHAN_TRIGGER_FRAC = 0.40
ORPHAN_TRIGGER_MED = 2
ORPHAN_HOST_MAX_FRAC = 0.005
ORPHAN_HOST_MIN = 50
ORPHAN_SCORE = 'rel'

def maxmod_lpa_candidate_pool(global_G, init_comm_dict, num_nodes, n_runs=None,
                              seed_base=None, collapse_frac=None, verbose=True,
                              n_workers=None, mode=None):
    from concurrent.futures import ThreadPoolExecutor

    n_runs = int(n_runs if n_runs is not None else LPA_CAND_RUNS)
    seed_base = int(seed_base if seed_base is not None else LPA_CAND_SEED_BASE)
    collapse_frac = float(collapse_frac) if collapse_frac is not None else 1.0
    mode = str(mode or LPA_CAND_MODE or 'both').lower()
    if mode not in ('scratch', 'refine', 'both'):
        mode = 'both'

    if n_workers is None:
        n_workers = min(max(n_runs, 1), 8)

    off = global_G._off
    cols = global_G._cols
    ws = global_G._ws

    init_labels = _dense_labels_from_comm_dict(init_comm_dict, num_nodes)

    if mode == 'refine':
        n_scratch, n_refine = 0, n_runs
    elif mode == 'scratch':
        n_scratch, n_refine = n_runs, 0
    else:
        n_refine = max(1, n_runs // 2)
        n_scratch = n_runs - n_refine

    if verbose:
        print(f"    [LPA-pool] mode={mode}: {n_scratch}x scratch + {n_refine}x refine, "
              f"{n_workers} threads")

    def _mk(kind, idxs):
        out = []
        for i in idxs:
            if kind == 'scratch':
                out.append(('scratch', (off, cols, ws, num_nodes, seed_base + i,
                                        collapse_frac, i)))
            else:
                out.append(('refine', (off, cols, ws, init_labels, num_nodes,
                                       seed_base + i, collapse_frac, i)))
        return out

    best = dict(mem=None, mod=-1.0, k=0, mx=0, kind=None, seed=None)
    runs = []

    def _run_batch(task_list):
        if not task_list:
            return []
        with ThreadPoolExecutor(max_workers=min(len(task_list), n_workers)) as executor:
            futures = [(kind, executor.submit(
                _lpa_single_run_csr if kind == 'scratch' else _lpa_single_run_csr_from_init,
                arg)) for kind, arg in task_list]
            for kind, future in futures:
                seed, k, mx, mod, status, mem_or_err = future.result()
                if status != 'ok':
                    runs.append((seed, k, mx, float('nan'), status, kind))
                    if verbose:
                        extra = (f" -> COLLAPSED (>{collapse_frac:.0%}N)"
                                 if status == 'collapsed' else f": {mem_or_err}")
                        print(f"    [LPA-cand] {kind:7s} seed={seed}: K={k:6d} "
                              f"max={mx:6d}{extra}")
                    continue
                runs.append((seed, k, mx, mod, 'ok', kind))
                if verbose:
                    print(f"    [LPA-cand] {kind:7s} seed={seed}: K={k:6d} "
                          f"max={mx:6d} mod={mod:.4f}")
                if mod > best['mod']:
                    best.update(mem=mem_or_err, mod=mod, k=k, mx=mx, kind=kind, seed=seed)

    half_s = max(1, n_scratch // 2) if n_scratch else 0
    half_r = max(1, n_refine // 2) if n_refine else 0
    seq_s = list(range(n_scratch))
    seq_r = list(range(n_scratch, n_scratch + n_refine))

    _run_batch(_mk('scratch', seq_s[:half_s]) + _mk('refine', seq_r[:half_r]))

    def _all_collapsed(kind):
        sub = [r for r in runs if r[5] == kind]
        return bool(sub) and all(r[4] == 'collapsed' for r in sub)

    rest_s = seq_s[half_s:] if not _all_collapsed('scratch') else []
    rest_r = seq_r[half_r:] if not _all_collapsed('refine') else []
    if rest_s or rest_r:
        _run_batch(_mk('scratch', rest_s) + _mk('refine', rest_r))
    elif verbose:
        print("    [LPA-pool] early-stop: remaining runs skipped (all collapsed)")

    ok_mods = [r[3] for r in runs if r[4] == 'ok']
    spread = (max(ok_mods) - min(ok_mods)) if len(ok_mods) >= 2 else 0.0

    info = dict(n_runs=n_runs, n_scratch=n_scratch, n_refine=n_refine, mode=mode,
                n_ok=len(ok_mods), mod_spread=spread,
                best_mod=best['mod'] if best['mem'] is not None else None,
                best_k=best['k'], best_max=best['mx'], best_kind=best['kind'],
                best_seed=best['seed'], runs=runs, collapse_frac=collapse_frac)
    return best['mem'], info

def maybe_apply_lpa_candidate(comm_dict, edge_df, num_nodes, dataname,
                              enabled=None, prebuilt_igraph=None, prebuilt_avg_deg=None,
                              prebuilt_edges=None, use_parallel=True,
                              global_G=None, use_csr=False,
                              use_refinement=True, mode=None):
    if enabled is None:
        enabled = ENABLE_LPA_CANDIDATE
    from collections import Counter as _Counter

    num_nodes = int(num_nodes)
    n_edges = int(len(edge_df))

    avg_deg = prebuilt_avg_deg if prebuilt_avg_deg is not None else (2.0 * n_edges / max(1, num_nodes))
    sizes = _Counter(comm_dict.values())
    cur_max = max(sizes.values()) if sizes else 0
    cur_k = len(sizes)
    max_frac = cur_max / max(1, num_nodes)

    size_scale = math.sqrt(2.0 * max(n_edges, 1))

    cap_frac = min(1.0, (float(LPA_CAND_MAX_GROWTH) * cur_max) / max(1, num_nodes))

    print(f"  [Stage G] N={num_nodes} E={n_edges} avg_deg={avg_deg:.2f} | "
          f"ASNCD K={cur_k} max={cur_max} ({max_frac:.4%} of N) | "
          f"pool cap={LPA_CAND_MAX_GROWTH:.1f}*cur_max={LPA_CAND_MAX_GROWTH * cur_max:.0f} "
          f"({cap_frac:.2%} of N); sqrt(2E)={size_scale:.0f} (diag only)")

    if not enabled:
        print("  [Stage G] disabled (--no-lpa-candidate) -> keep pipeline output")
        return comm_dict, dict(applied=False, reason='disabled')

    c1 = num_nodes >= LPA_CAND_MIN_N
    c2 = avg_deg < LPA_CAND_MAX_DEG
    print(f"  [Stage G] trigger: N>={LPA_CAND_MIN_N}?{c1}  "
          f"avg_deg<{LPA_CAND_MAX_DEG}?{c2}  "
          f"(diag: max_frac={max_frac:.4%})")

    required = c1 and c2
    if not required:
        return comm_dict, dict(applied=False, reason='not-triggered',
                               c1=c1, c2=c2,
                               avg_deg=avg_deg, max_frac=max_frac)

    print(f"  [Stage G] TRIGGERED (heavy-tail missing) -> running "
          f"{LPA_CAND_RUNS}x LPA with modularity selection")
    _t = time.time()

    if use_csr and global_G is not None:
        print(f"  [Stage G] using LightweightGraph CSR structure for LPA (zero overhead)")

        try:
            _w = global_G._ws
            if _w is not None and _w.size:
                print(f"  [Stage G] CSR weights: min={float(_w.min()):.4f} "
                      f"max={float(_w.max()):.4f} mean={float(_w.mean()):.4f}")
        except Exception:
            pass
        mem_lpa, info = maxmod_lpa_candidate_pool(global_G, comm_dict, num_nodes,
                                                  n_runs=LPA_CAND_RUNS, mode=mode,
                                                  collapse_frac=cap_frac)

        mem_asncd = comm_dict_to_membership(comm_dict, num_nodes)
        mod_asncd = _compute_modularity_numpy(global_G._off, global_G._cols, global_G._ws,
                                              np.array(mem_asncd), num_nodes)

    elif prebuilt_edges is not None and use_parallel:
        print(f"  [Stage G] using prebuilt edges for parallel LPA: E={prebuilt_edges.shape[0]}")

        mem_lpa, info = maxmod_lpa_candidate_parallel(prebuilt_edges, num_nodes,
                                                      collapse_frac=cap_frac)

        import igraph as ig
        g_tmp = ig.Graph(n=num_nodes, edges=prebuilt_edges.tolist(), directed=False)
        mem_asncd = comm_dict_to_membership(comm_dict, num_nodes)
        mod_asncd = float(g_tmp.modularity(mem_asncd))
    elif prebuilt_igraph is not None:
        g = prebuilt_igraph
        print(f"  [Stage G] using prebuilt igraph: N={g.vcount()} E={g.ecount()}")
        mem_asncd = comm_dict_to_membership(comm_dict, num_nodes)
        mod_asncd = float(g.modularity(mem_asncd))
        mem_lpa, info = maxmod_lpa_candidate(g, num_nodes, collapse_frac=cap_frac)
    else:
        g = build_modularity_graph(edge_df, num_nodes)
        print(f"  [Stage G] igraph built (no prebuild): N={g.vcount()} E={g.ecount()}")
        mem_asncd = comm_dict_to_membership(comm_dict, num_nodes)
        mod_asncd = float(g.modularity(mem_asncd))
        mem_lpa, info = maxmod_lpa_candidate(g, num_nodes, collapse_frac=cap_frac)

    print(f"  [Stage G] graph ready ({time.time() - _t:.1f}s); ASNCD modularity={mod_asncd:.4f}")
    if info.get('mod_spread') is not None:
        print(f"  [Stage G] candidate pool: n_ok={info.get('n_ok')}/{info.get('n_runs')}, "
              f"mod spread={info['mod_spread']:.6f}"
              f"{'  (DEGENERATE)' if info['mod_spread'] < 1e-4 else ''}")
    info.update(avg_deg=avg_deg, max_frac=max_frac, cur_k=cur_k, cur_max=cur_max,
                mod_asncd=mod_asncd, elapsed=time.time() - _t)

    if mem_lpa is None:
        print("  [Stage G] no valid LPA candidate -> keep pipeline output")
        return comm_dict, dict(applied=False, reason='no-candidate', **info)

    mod_lpa = info['best_mod']
    print(f"  [Stage G] best LPA [{info.get('best_kind')}/seed={info.get('best_seed')}]: "
          f"K={info['best_k']} max={info['best_max']} mod={mod_lpa:.4f}  "
          f"vs ASNCD K={cur_k} max={cur_max} mod={mod_asncd:.4f} "
          f"(delta={mod_lpa - mod_asncd:+.4f})")

    if (mod_lpa - mod_asncd) <= 0:
        print(f"  [Stage G] mod gain {mod_lpa - mod_asncd:+.4f} <= 0 "
              f"-> keep pipeline output")
        return comm_dict, dict(applied=False, reason='no-mod-gain', **info)

    if LPA_CAND_SHAPE_GUARD:
        k_ratio = info['best_k'] / max(1, cur_k)
        tail_ok = info['best_max'] > cur_max

        growth = info['best_max'] / max(1, cur_max)
        print(f"  [Stage G] size guard: candidate max={info['best_max']} / "
              f"cur_max={cur_max} = {growth:.2f}x (limit {LPA_CAND_MAX_GROWTH}); "
              f"diag {info['best_max']/max(1.0, size_scale):.1f}x sqrt(2E)")
        if growth > LPA_CAND_MAX_GROWTH:
            print(f"  [Stage G] REJECT: largest community inflates x{growth:.2f} "
                  f"(> {LPA_CAND_MAX_GROWTH}) — LPA giant-blob failure -> keep pipeline")
            return comm_dict, dict(applied=False, reason='shape-growth', **info)
        if not tail_ok:

            print("  [Stage G] WARN: candidate tail not heavier than ASNCD "
                  "(heavy-tail premise not met) -> still allowed")

    print(f"  [Stage G] APPLIED: replacing partition with Max-Mod LPA "
          f"(K {cur_k} -> {info['best_k']}, max {cur_max} -> {info['best_max']})")
    new_comm_dict = {v: mem_lpa[v] for v in range(num_nodes)}
    return new_comm_dict, dict(applied=True, reason='applied', **info)

def evaluate_with_correct_format(true_comms, comm_dict, all_nodes, name):

    pred_comms = convert_comm_dict_to_comms_list(comm_dict)

    gt_node_set = set().union(*true_comms) if true_comms else set()

    cleaned_pred_comms = []
    original_node_count = sum(len(c) for c in pred_comms)

    for comm in pred_comms:
        filtered_comm = set(comm) & gt_node_set
        if filtered_comm:
            cleaned_pred_comms.append(filtered_comm)

    cleaned_node_count = sum(len(c) for c in cleaned_pred_comms)
    print(f"  [Node Filter] Pred nodes: {original_node_count} -> {cleaned_node_count} | "
          f"Pred comms: {len(pred_comms)} -> {len(cleaned_pred_comms)}")

    nmi = xmeasures.nmi(cleaned_pred_comms, true_comms, name)
    f1 = xmeasures.f1(cleaned_pred_comms, true_comms, name)

    def _to_float(s):
        s = (s or "").strip()
        s = s.split()[0] if s else "0"
        try:
            return float(s)
        except ValueError:
            return 0.0

    nmi_val = _to_float(nmi)
    f1_val = _to_float(f1)

    import re as _re
    prc_val, rec_val = None, None
    try:
        _m = _re.search(r"Prc:\s*([0-9.]+)", f1 or "")
        if _m:
            prc_val = float(_m.group(1))
        _m = _re.search(r"Rec:\s*([0-9.]+)", f1 or "")
        if _m:
            rec_val = float(_m.group(1))
    except Exception:
        pass

    if prc_val is not None and rec_val is not None:
        print(f"  NMI={nmi_val:.4f}  F1={f1_val:.4f}  (Prc={prc_val:.4f}, Rec={rec_val:.4f})")
    else:
        print(f"  NMI={nmi_val:.4f}  F1={f1_val:.4f}")

    return nmi_val, f1_val

def load_data(edge_file_path, comm_file_path):

    with open(comm_file_path) as f:
        communities = [[int(i) for i in x.split()] for x in f]

    _edf = pd.read_csv(edge_file_path, sep=r"\s+", header=None, dtype=np.int64,
                       comment="#")
    _u = _edf[0].values
    _v = _edf[1].values
    _ne = _u != _v
    _u = _u[_ne]
    _v = _v[_ne]
    _lo = np.minimum(_u, _v)
    _hi = np.maximum(_u, _v)
    _uniq = np.unique(np.concatenate([_lo, _hi]))
    _ei = np.searchsorted(_uniq, _lo)
    _ej = np.searchsorted(_uniq, _hi)

    edges = np.stack([_ei, _ej], axis=1)
    mapping = {int(n): i for i, n in enumerate(_uniq.tolist())}

    communities = [[mapping[node] for node in com] for com in communities]

    num_node = len(_uniq)
    num_edges = len(edges)
    num_comm = len(communities)

    print(f"[{os.path.basename(edge_file_path).upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")

    new_nodes = list(range(num_node))

    return num_node, num_edges, num_comm, new_nodes, edges, communities

"""ASNCD 主流水线（无监督）。

        流水线共 5 个阶段，各阶段均以 GT-free 判据守卫，默认对所有数据集同式生效：
            Stage 0  分块（METIS/连通性）+ 块内 CPM + 归属块写回 → 初始划分
            Stage 1  SE 博弈全局优化（节点级结构熵迁移）
            Stage 2  结构自适应小社区回收（CPM 增益判据）+ 互最优邻居合并
            Stage 3  巨社区重建（SGM）+ 过度合并社区定点分裂（SG-Split）
            Stage 4  重尾缺失处理（Max-Mod LPA 候选通道）
            Stage 5  孤儿微社区宿主归并

        resolution_scale 可为数值，或字符串 'auto'（结构化自适应，
        见 get_adaptive_resolution_scale）；'auto' 时实际选值写入 rs_out。
"""

def execute_ASNCD_pipeline_unsupervised(edge_file_path, comm_file_path, dataname, network_type,
                                        partition_method='bfs', resolution_scale=0.8,
                                        merge_ratio=0.10,

                                        merge_score_mult=None, merge_cap=None,
                                        merge_cpm=False,
                                        merge_cpm_iter=False, merge_gamma_mult=2.0,
                                        enable_se_game=True, enable_merge=True,
                                        stage_eval=False, save_pred=False,
                                        save_pred_dir=os.environ.get('ASNCD_PRED_DIR', 'E:/HIDC-UD/preds'),

                                        rs_out=None):

    global MERGE_DUMP_TAG
    MERGE_DUMP_TAG = str(dataname)

    start_total_time1 = time.time()
    memory_snapshot("startup")

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)

    num_node, num_edges, num_comm, all_nodes, edges, communities = load_data(
        edge_file_path, comm_file_path
    )
    edge_df = pd.DataFrame(edges, columns=['u', 'v'])
    memory_snapshot("after load data")

    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    _n_nodes = len(all_nodes)
    u_degrees = np.bincount(edge_df['u'].values, minlength=_n_nodes)
    v_degrees = np.bincount(edge_df['v'].values, minlength=_n_nodes)
    node_degree = (u_degrees + v_degrees).astype(np.float32)

    node_degree_dict = dict(zip(all_nodes, node_degree))

    total_time1 = (time.time() - start_total_time1) / 60
    print(f"\nTotal time: {total_time1:.2f} minutes")
    memory_snapshot("after degree computation")

    block_size, MIN_COMM_SIZE, cn_base_alpha = get_adaptive_params(len(all_nodes), len(edge_df))
    if BLOCK_SIZE_OVERRIDE is not None:
        block_size = int(BLOCK_SIZE_OVERRIDE)
    if MIN_COMM_SIZE_OVERRIDE is not None:
        MIN_COMM_SIZE = int(MIN_COMM_SIZE_OVERRIDE)

    if isinstance(resolution_scale, str) and resolution_scale == 'auto':
        _t_rs = time.time()
        frac10core, lcc_frac = compute_structure_signals(len(all_nodes), edge_df)
        resolution_scale = get_adaptive_resolution_scale(frac10core, lcc_frac)
        if rs_out is not None:
            rs_out.update(dict(resolution_scale=resolution_scale,
                               frac10core=frac10core, lcc_frac=lcc_frac,
                               rs_feature_sec=time.time() - _t_rs))
        print(f"  [RS-adaptive] frac10core={frac10core:.3f}, lcc_frac={lcc_frac:.3f} "
              f"→ resolution_scale={resolution_scale}")

    print(f"  block_size: {block_size}, MIN_COMM_SIZE: {MIN_COMM_SIZE}, cn_base_alpha: {cn_base_alpha}")

    klist = [block_size]

    for block_size in klist:
        start_total_time = time.time()

        print(f"  block_size: {block_size}, MIN_COMM_SIZE: {MIN_COMM_SIZE}, cn_base_alpha: {cn_base_alpha}")
        print(f"Block size: {block_size}")
        start_cpp = time.time()

        print(f"Partition method: {partition_method}")
        if partition_method == 'metis':
            base_blocks, new_all_nodes = split_data_by_metis_optimized(
                edge_df, all_nodes, node_degree_dict, block_size
            )
        else:
            base_blocks, new_all_nodes = split_data_by_connectivity(
                edge_df, all_nodes, node_degree_dict, block_size
            )

        final_blocks = []
        home_sets = []
        num_blocks = len(base_blocks)

        _nodes_np = np.fromiter(all_nodes, dtype=np.int64, count=len(all_nodes))
        _identity = bool(len(all_nodes) and
                         np.array_equal(_nodes_np, np.arange(len(all_nodes))))
        if _identity:
            home_bid = np.full(len(all_nodes), -1, dtype=np.int64)
            for bid, bnodes in enumerate(base_blocks):
                home_bid[np.asarray(bnodes, dtype=np.int64)] = bid
            _eu_m = np.asarray(edge_df['u'].values, dtype=np.int64)
            _ev_m = np.asarray(edge_df['v'].values, dtype=np.int64)

        for i in range(num_blocks):
            home = list(base_blocks[i])
            home_sets.append(set(home))

            final_blocks.append(home)

        blocks = final_blocks

        if _identity:

            _bb_u = home_bid[_eu_m]
            _bb_v = home_bid[_ev_m]
            cross_block_edges = int(np.count_nonzero(_bb_u != _bb_v))
            total_edges = int(_bb_u.size)
            del _bb_u, _bb_v

        else:
            node_to_block = {}
            for bid, block_nodes in enumerate(blocks):
                for n in block_nodes:
                    node_to_block[n] = bid
            cross_block_edges = 0
            total_edges = 0
            for u, v in edges:
                total_edges += 1
                if node_to_block.get(u) != node_to_block.get(v):
                    cross_block_edges += 1
            cross_block_edges = int(cross_block_edges)
            total_edges = int(total_edges)

        print(f"Cross-block edges: {cross_block_edges}/{total_edges} "
            f"({cross_block_edges/total_edges:.1%})")

        end_cpp = (time.time() - start_cpp) / 60
        print(f"\nBFS partitioning time: {end_cpp:.2f} minutes")
        print(f"Generated {len(blocks)} blocks")
        memory_snapshot("after BFS partitioning")

        num_blocks = len(blocks)

        if num_blocks == 0:
            raise ValueError("Data splitting failed")

        _t_prefilter = time.time()
        _nb = len(blocks)
        if _identity:
            _eu_pb = _eu_m
            _ev_pb = _ev_m
            _home_bid = home_bid
            _hu = _home_bid[_eu_pb]
            _hv = _home_bid[_ev_pb]
            _rows = np.arange(_eu_pb.size, dtype=np.int64)
            _mSame = _hu == _hv

            _kb_list = [_hu[_mSame]]
            _kr_list = [_rows[_mSame]]
            _kb = np.concatenate(_kb_list)
            _kr = np.concatenate(_kr_list)
            del _kb_list, _kr_list
            _mvalid = (_kb >= 0) & (_kb < _nb)
            _kb = _kb[_mvalid]
            _kr = _kr[_mvalid]
            _ord = np.lexsort((_kr, _kb))
            _kb = _kb[_ord]
            _kr = _kr[_ord]
            del _ord, _mvalid, _rows, _mSame

            _cnts = np.bincount(_kb, minlength=_nb)
            _offs = np.zeros(_nb + 1, dtype=np.int64)
            np.cumsum(_cnts, out=_offs[1:])
            _deg_np = node_degree
            pre_block_args = []
            for _bid, _bnodes in enumerate(blocks):
                _s = int(_offs[_bid])
                _e = int(_offs[_bid + 1])
                _idx = _kr[_s:_e]
                if _idx.size:
                    _sub = edge_df.iloc[_idx]
                    _sub_nodes = np.union1d(_sub['u'].values, _sub['v'].values)
                    _deg_sub = {int(n): float(_deg_np[n]) for n in _sub_nodes}
                else:
                    _sub = edge_df.iloc[0:0]
                    _deg_sub = {}
                pre_block_args.append((_bid, _sub, list(_bnodes), cn_base_alpha,
                                       _deg_sub))
            del _kb, _kr, _cnts, _offs, _hu, _hv
            memory_snapshot("after prefilter numpy group")
        else:
            _u_all = edge_df['u'].values
            _v_all = edge_df['v'].values
            _rows_per_block = [[] for _ in range(_nb)]
            _node_blocks = defaultdict(list)
            for _bid, _bnodes in enumerate(blocks):
                for _n in _bnodes:
                    _node_blocks[_n].append(_bid)
            for _row_i, (_uu, _vv) in enumerate(zip(_u_all.tolist(), _v_all.tolist())):
                _bu = _node_blocks.get(_uu)
                _bv = _node_blocks.get(_vv)
                if not _bu or not _bv:
                    continue
                if len(_bu) <= len(_bv):
                    _small, _large = _bu, _bv
                else:
                    _small, _large = _bv, _bu
                for _b in _small:
                    if _b in _large:
                        _rows_per_block[_b].append(_row_i)
            pre_block_args = []
            for _bid, _bnodes in enumerate(blocks):
                _idx = np.asarray(_rows_per_block[_bid], dtype=np.int64)
                _sub = edge_df.iloc[_idx] if _idx.size else edge_df.iloc[0:0]
                _sub_nodes = set(_sub['u'].tolist()) | set(_sub['v'].tolist()) if len(_sub) else set()
                _deg_sub = {n: node_degree_dict.get(n, 1.0) for n in _sub_nodes}
                pre_block_args.append((_bid, _sub, list(_bnodes), cn_base_alpha, _deg_sub))
            del _rows_per_block, _node_blocks
        print(f"Block edge pre-filter time: {time.time() - _t_prefilter:.2f}s")

        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.imap_unordered(process_block_prefiltered, pre_block_args)

            weighted_edge_dict = {}
            for bid, bedge in results:
                weighted_edge_dict[bid] = bedge

        weighted_edge_list = []
        for i in sorted(weighted_edge_dict.keys()):
            df = weighted_edge_dict[i]
            if not df.empty and len(df) > 0:
                weighted_edge_list.append(df)

        if weighted_edge_list:
            weighted_edge_df = pd.concat(weighted_edge_list, ignore_index=True)
            weighted_edge_df = weighted_edge_df.astype({
                'u': int,
                'v': int,
                'weight': float
            })
        else:
            weighted_edge_df = pd.DataFrame(columns=['u', 'v', 'weight'])

        print(f"Summarized edge weights: {len(weighted_edge_df)} edges")

        memory_snapshot("after edge weight computation")

        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)
        for f in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, f))

        print(f"\n[5/6 Using Leiden for block community detection...]")

        comm_args = []
        if _identity:
            _wu = np.asarray(weighted_edge_df['u'].values, dtype=np.int64)
            _wv = np.asarray(weighted_edge_df['v'].values, dtype=np.int64)
            _hu2 = home_bid[_wu]
            _hv2 = home_bid[_wv]
            _wrows = np.arange(_wu.size, dtype=np.int64)
            _cand_b = np.concatenate([_hu2, _hv2])
            _cand_r = np.concatenate([_wrows, _wrows])
            _ok = (_cand_b >= 0) & (_cand_b < _nb)
            _cand_b = _cand_b[_ok]
            _cand_r = _cand_r[_ok]
            _wo = np.lexsort((_cand_r, _cand_b))
            _cand_b = _cand_b[_wo]
            _cand_r = _cand_r[_wo]

            _ck = _cand_b * _wrows.size + _cand_r
            _cu, _cinv = np.unique(_ck, return_index=True)
            _cu = _cand_b[np.asarray(_cinv).ravel()]
            _cw = _cand_r[np.asarray(_cinv).ravel()]
            del _ck, _cand_b, _cand_r, _wo
            _wcnt = np.bincount(_cu, minlength=_nb)
            _woffs = np.zeros(_nb + 1, dtype=np.int64)
            np.cumsum(_wcnt, out=_woffs[1:])
            for _bid, _bnodes in enumerate(blocks):
                _s = int(_woffs[_bid])
                _e = int(_woffs[_bid + 1])
                block_edges = weighted_edge_df.iloc[_cw[_s:_e]] \
                    if _e > _s else weighted_edge_df.iloc[0:0]
                comm_args.append((block_edges, _bnodes, _bid, resolution_scale))
            del _wu, _wv, _hu2, _hv2, _wrows, _cu, _cw
            memory_snapshot("after per-block numpy filter")
        else:
            for block_id, block_nodes in enumerate(blocks):
                block_edges = weighted_edge_df[
                    weighted_edge_df['u'].isin(block_nodes) |
                    weighted_edge_df['v'].isin(block_nodes)
                ]
                comm_args.append((block_edges, block_nodes, block_id, resolution_scale))

        block_results = []
        with Pool(processes=NUM_PROCESSES) as pool:
            comm_results = pool.imap_unordered(generate_block_community, comm_args)
            for result in comm_results:
                block_results.append(result)

        block_sizes = {bid: len(blocks[bid]) for bid in range(len(blocks))}
        block_results.sort(key=lambda x: (block_sizes[x[0]], x[0]))

        global_comm_dict = {}
        for bid, bcomm in block_results:
            home = home_sets[bid]
            for n, c in bcomm.items():
                if n in home:
                    global_comm_dict[n] = c
        memory_snapshot("after block community detection")

        describe_partition(global_comm_dict, "S0_CPM", communities)

        global_G = build_lightweight_graph(edge_df, all_nodes)
        memory_snapshot("after global graph build")

        start_new = time.time()

        if enable_se_game:
            se_init_partition = global_comm_dict
            if SE_COLD_START:

                se_init_partition = {n: n for n in all_nodes}
                print(f"  [SE Game v2] COLD START（--se-cold-start）：{len(all_nodes)} 个 "
                      f"singleton 起步（对照 warm：{len(set(global_comm_dict.values()))} 社区）")
            final_comm_dict1 = global_optimization_se_game_v2(
                global_G,
                se_init_partition,
                all_nodes,

                verbose=True,
            )
        else:
            if True:
                print("  [SE Game v2] DISABLED (--no-se-game)：跳过全局博弈，"
                      "直接使用块内 CPM 划分")
            final_comm_dict1 = dict(global_comm_dict)

        describe_partition(final_comm_dict1, "S1_SEgame", communities)

        if stage_eval and communities:
            n1, f1_ = evaluate_with_correct_format(communities, final_comm_dict1, all_nodes,
                                                   f"{dataname}_stage1_segame")
            print(f"  [STAGE 1 / after SE game] NMI={n1:.4f} F1={f1_:.4f} "
                  f"#comms={len(set(final_comm_dict1.values()))}")

        if MERGE_DUMP_DIR:
            try:
                os.makedirs(MERGE_DUMP_DIR, exist_ok=True)
                with open(os.path.join(MERGE_DUMP_DIR,
                                       f"{MERGE_DUMP_TAG}_pre.pkl"), 'wb') as _f:
                    pickle.dump({'dataset': MERGE_DUMP_TAG,
                                 'comm_dict': final_comm_dict1,
                                 'communities': communities,
                                 'all_nodes': all_nodes}, _f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as _e:
                print(f"  [MergeDump] FAILED (pre-partition): {_e}")

        final_comm_dict = dict(final_comm_dict1)

        if ENABLE_CPM_CLEANUP:
            final_comm_dict, n_cpm = merge_small_communities_cpm(
                final_comm_dict, edge_df, MIN_COMM_SIZE, resolution_scale,
                len(all_nodes), G=global_G)
            print(f"  [CPMCleanup] τ={MIN_COMM_SIZE}, merged {n_cpm} small comms "
                  f"→ {len(set(final_comm_dict.values()))} comms")

        if ENABLE_MB:
            final_comm_dict = merge_communities_mutual_best(final_comm_dict, global_G)
            memory_snapshot("after mutual-best merge")

        if ENABLE_SGM:
            final_comm_dict, giant_nodes_first = rebuild_giant_communities(
                final_comm_dict, global_G, len(all_nodes),
                total_edges=len(edge_df))
            memory_snapshot("after giant community rebuild")

        if ENABLE_SG_SPLIT:
            final_comm_dict, _n_sp = split_glued_communities(
                final_comm_dict, edge_df, len(all_nodes),
                resolution_scale)
            if _n_sp:
                memory_snapshot("after glued split")

        nmi_val, f1_val = None, None
        if communities:

            final_comm_dict, _lpa_info = maybe_apply_lpa_candidate(
                final_comm_dict, edge_df, len(all_nodes), dataname,
                global_G=global_G, use_csr=True, mode=LPA_CAND_MODE)
            if _lpa_info.get('applied'):
                memory_snapshot("after LPA candidate (Stage G)")
            metrics = evaluate_with_correct_format(communities, final_comm_dict,all_nodes,dataname)
            nmi_val, f1_val = metrics

            _save_dir = os.environ.get("ASNCD_PRED_DIR", "")
            if save_pred and _save_dir:
                os.makedirs(_save_dir, exist_ok=True)
                _pl = defaultdict(list)
                for _n, _c in final_comm_dict.items():
                    _pl[_c].append(_n)
                _p = os.path.join(_save_dir, f"{dataname}_pred.txt")
                with open(_p, "w", encoding="utf-8") as _f:
                    for _c, _nodes in _pl.items():
                        _f.write(" ".join(str(int(x)) for x in sorted(_nodes)) + "\n")
                print(f"  [SavePred] {_p} ({len(_pl)} comms)")

        total_time = (time.time() - start_total_time) / 60
        print(f"\nTotal time: {total_time + total_time1:.2f} minutes")

    return nmi_val, f1_val

SE_CUT_CONVENTION = 'legacy'

SE_STOP_MODE = 'trans_frac'

def _build_se_kernel():
    try:
        from numba import njit
        import math as _m

        @njit(cache=True)
        def _se_round(off, cols, ws, node_deg, comm, vol, cut, order,
                      mark, touched, deg_in, tag0, gv):
            n = order.shape[0]
            delta_sum = 0.0
            n_trans = 0
            n_stay = 0
            for k in range(n):
                i = order[k]
                c_src = comm[i]
                if c_src < 0:
                    continue
                deg_i = node_deg[i]
                if deg_i == 0.0:
                    continue
                tag = tag0 + k
                nt = 0
                s = off[i]
                e = off[i + 1]
                for p in range(s, e):
                    j = cols[p]
                    cj = comm[j]
                    if cj < 0:
                        continue
                    w = ws[p]
                    if mark[cj] != tag:
                        mark[cj] = tag
                        touched[nt] = cj
                        nt += 1
                        deg_in[cj] = w
                    else:
                        deg_in[cj] += w

                vol_s = vol[c_src]
                cut_s = cut[c_src]
                deg_i_src = deg_in[c_src] if mark[c_src] == tag else 0.0
                if abs(vol_s - deg_i) < 1e-8:
                    dl_src = 0.0
                else:
                    src_v = vol_s
                    tgt_v = vol_s - deg_i
                    src_c = cut_s
                    tgt_c = cut_s + 2.0 * deg_i_src - deg_i
                    dl_src = ((tgt_c / gv) * _m.log(tgt_v / gv)
                              - (src_c / gv) * _m.log(src_v / gv)
                              + (deg_i / gv) * _m.log(src_v / gv)
                              + (tgt_v / gv) * _m.log(src_v / tgt_v))
                delta_max = dl_src
                best = c_src
                best_in = deg_i_src
                for t in range(nt):
                    c_tgt = touched[t]
                    if c_tgt == c_src:
                        continue
                    deg_i_tgt = deg_in[c_tgt]

                    src_v2 = vol[c_tgt] + deg_i
                    tgt_v2 = vol[c_tgt]
                    src_c2 = cut[c_tgt] - 2.0 * deg_i_tgt + deg_i
                    tgt_c2 = cut[c_tgt]
                    dl_tgt = ((tgt_c2 / gv) * _m.log(tgt_v2 / gv)
                              - (src_c2 / gv) * _m.log(src_v2 / gv)
                              + (deg_i / gv) * _m.log(src_v2 / gv)
                              + (tgt_v2 / gv) * _m.log(src_v2 / tgt_v2))
                    dt = dl_src - dl_tgt
                    if dt > delta_max:
                        delta_max = dt
                        best = c_tgt
                        best_in = deg_i_tgt
                if best != c_src and delta_max > 1e-10:
                    vol[c_src] -= deg_i
                    if vol[c_src] > 0.0:
                        cut[c_src] += 2.0 * deg_i_src - deg_i
                    else:
                        cut[c_src] = 0.0
                    comm[i] = best
                    vol[best] += deg_i
                    cut[best] -= 2.0 * best_in - deg_i
                    delta_sum += delta_max
                    n_trans += 1
                else:
                    n_stay += 1
            return delta_sum, n_trans, n_stay

        return _se_round
    except Exception:
        return None

_SE_ROUND = _build_se_kernel()

def _se_game_v2_fast(G, initial_comm_dict, all_nodes,
                      verbose):
    n_T = G.n_nodes
    node_deg = G.weighted_degrees()
    graph_vol = float(node_deg.sum())
    if graph_vol == 0.0:
        return dict(initial_comm_dict)

    N = len(all_nodes)
    n_d = len(initial_comm_dict)
    labels, dense = np.unique(
        np.fromiter(initial_comm_dict.values(), dtype=np.int64, count=n_d),
        return_inverse=True)
    dense = np.asarray(dense).ravel()
    idxs = np.fromiter((G.node_to_idx[n] for n in initial_comm_dict.keys()),
                       dtype=np.int64, count=n_d)
    K = int(labels.size)
    comm = np.full(n_T, -1, dtype=np.int64)
    comm[idxs] = dense

    comm_vol = np.bincount(dense, weights=node_deg[idxs], minlength=K)

    vw = G.edge_weights
    ci = comm[vw._iu]
    cj = comm[vw._iv]
    mask = (ci >= 0) & (cj >= 0) & (ci != cj)
    comm_cut = (np.bincount(ci[mask], weights=vw._w[mask], minlength=K)
                + np.bincount(cj[mask], weights=vw._w[mask], minlength=K))
    if SE_CUT_CONVENTION == 'legacy':
        comm_cut = comm_cut * 0.5

    pos = node_deg[node_deg > 0] / graph_vol
    se_1d = float(-np.sum(pos * np.log(pos)))
    avg_se_1d = se_1d / max(N, 1)

    if verbose:
        print(f"  [SE Game v2] Init: {K} comms, SE_1d={se_1d:.4f}, "
              f"avg_SE={avg_se_1d:.6f}  (fast/numba)")

    order = np.argsort(-node_deg, kind='stable').astype(np.int64)
    max_deg = int(np.max(np.diff(G._off))) if n_T else 1
    mark = np.zeros(K, dtype=np.int64)
    touched = np.zeros(max(max_deg, 1), dtype=np.int64)
    deg_in = np.zeros(K, dtype=np.float64)

    _stop_thr = 1.0
    if SE_STOP_MODE == 'trans_frac':
        _stop_thr = max(1.0, float(N) / math.sqrt(graph_vol))
    if verbose:
        print(f"  [SE Game v2] stop_mode={SE_STOP_MODE}, "
              f"trans_thr={_stop_thr:.1f} (N={N}, sqrt(2E)={math.sqrt(graph_vol):.0f})")

    iteration = 0
    while True:
        delta_se_sum, num_trans, num_stay = _SE_ROUND(
            G._off, G._cols, G._ws, node_deg, comm, comm_vol, comm_cut,
            order, mark, touched, deg_in, iteration * N + 1, graph_vol)

        if verbose:
            print(f"  [SE Game v2] Iter {iteration+1}: "
                  f"ΔSE={delta_se_sum:.6f}, #trans={num_trans}, #stay={num_stay}")

        if iteration == 0:
            delta_se_first = max(delta_se_sum, 1e-12)

        if num_trans < _stop_thr:
            if verbose:
                print(f"  [SE Game v2] Converged: #trans={num_trans} < "
                      f"thr={_stop_thr:.1f} after {iteration + 1} iter(s)")
            break

        if delta_se_sum <= 0:
            if verbose:
                print(f"  [SE Game v2] Converged: ΔSE={delta_se_sum:.6e} <= 0")
            break

        iteration += 1

    return {n: int(labels[comm[G.node_to_idx[n]]]) for n in initial_comm_dict}

def global_optimization_se_game_v2(G, initial_comm_dict, all_nodes,

                                   verbose=True):

    if not initial_comm_dict:
        return dict(initial_comm_dict)
    if not (hasattr(G, 'weighted_degrees') and hasattr(G, 'edge_weights')):
        raise TypeError("G 缺少 weighted_degrees/edge_weights，无法使用 fast 路径。")
    return _se_game_v2_fast(G, initial_comm_dict, all_nodes,
                                 verbose)

def process_block_prefiltered(args):
    block_id, block_edge_df, block_nodes, cn_base_alpha, deg_sub = args
    return calc_block_edge_weight_no_queue(block_edge_df, block_nodes, block_id,
                                           cn_base_alpha, deg_sub)

DATASET_CONFIGS = {

    'lfr_a_mu01': {
        'edge_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_a_mu01/graph.ungraph.txt',
        'community_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_a_mu01/gt.cmty.txt',
        'description': 'LFR n=5000 mu=0.1 max_comm=3.9%N', 'network_type': 'synthetic',
    },
    'lfr_a_mu03': {
        'edge_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_a_mu03/graph.ungraph.txt',
        'community_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_a_mu03/gt.cmty.txt',
        'description': 'LFR n=5000 mu=0.3 max_comm=3.9%N', 'network_type': 'synthetic',
    },
    'lfr_a_mu05': {
        'edge_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_a_mu05/graph.ungraph.txt',
        'community_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_a_mu05/gt.cmty.txt',
        'description': 'LFR n=5000 mu=0.5 max_comm=3.9%N', 'network_type': 'synthetic',
    },
    'lfr_b_mu03': {
        'edge_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_b_mu03/graph.ungraph.txt',
        'community_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_b_mu03/gt.cmty.txt',
        'description': 'LFR n=20000 mu=0.3 max_comm=2.0%N', 'network_type': 'synthetic',
    },
    'lfr_giant_in': {
        'edge_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_giant_in/graph.ungraph.txt',
        'community_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_giant_in/gt.cmty.txt',
        'description': 'LFR n=10000 mu=0.2 giant 12.1%N（SGM 窗口内）', 'network_type': 'synthetic',
    },
    'lfr_giant_out': {
        'edge_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_giant_out/graph.ungraph.txt',
        'community_path': 'C:/Users/Administrator/WorkBuddy/2026-11-04-18-13-47/exp_w2/datasets/lfr_giant_out/gt.cmty.txt',
        'description': 'LFR n=10000 mu=0.2 giant 25.8%N（超出 SGM_FRAC_MAX=0.20）', 'network_type': 'synthetic',
    },
    'facebook': {
        'edge_path': 'dataset/facebook-1.90.ungraph.txt',
        'community_path': 'dataset/facebook-1.90.cmty.txt',
        'description': 'Facebook social network',
        'network_type': 'social',

    },
    'amazon1': {
        'edge_path': 'dataset/amazon-1.90.ungraph.txt',
        'community_path': 'dataset/amazon-1.90.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase',

    },
    'lj1': {
        'edge_path': 'dataset/lj-1.90.ungraph.txt',
        'community_path': 'dataset/lj-1.90.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social',

    },
    'dblp1': {
        'edge_path': 'dataset/dblp-1.90.ungraph.txt',
        'community_path': 'dataset/dblp-1.90.cmty.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration',

    },
    'dblp2': {
        'edge_path': 'dataset/dblp.ungraph.txt',
        'community_path': 'dataset/dblp_communities.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration',

    },
    'amazon2': {
        'edge_path': 'dataset/com-amazon.ungraph.txt',
        'community_path': 'dataset/com-amazon.all.dedup.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase',

    },
    'lj2': {
        'edge_path': 'dataset/lj.ungraph.txt',
        'community_path': 'dataset/lj.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social',

    },
    'lj3': {
        'edge_path': 'dataset/com-lj.ungraph.txt',
        'community_path': 'dataset/com-lj.all.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social',

    },
    'friend': {
        'edge_path': 'dataset/com-friendster.ungraph.txt',
        'community_path': 'dataset/com-friendster.all.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social',

    },
    'youtube': {
        'edge_path': 'dataset/com-youtube.ungraph.txt',
        'community_path': 'dataset/com-youtube.all.cmty.txt',
        'description': 'YouTube social network',
        'network_type': 'social',

    },
    'twitter': {
        'edge_path': 'dataset/twitter-1.90.ungraph.txt',
        'community_path': 'dataset/twitter-1.90.cmty.txt',
        'description': 'Twitter social network (held-out zero-shot)',
        'network_type': 'social'
    },

}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASNCD s8: 统一超参数网格搜索 / 最终验证")

    parser.add_argument('--datasets', type=str,
                            default='facebook,amazon1,lj1,dblp1,amazon2,dblp2,lj2,youtube,lj3',
                            help='逗号分隔的数据集列表')

    parser.add_argument('--merge-score-mult', type=float, default=0.0,
                        help='自适应合并判据：score >= score_mult*median(score)；'
                             '>0 时替代 merge_ratio 配额（默认 0 = 关闭，走配额）')
    parser.add_argument('--merge-cap', type=float, default=-1.0,
                        help='自适应模式下的合并对比例上限（安全帽），<0 表示不限制')
    parser.add_argument('--merge-cpm', action='store_true', default=False,
                        help='自适应合并判据改用 CPM 增益：w_ab > gamma*|a|*|b|，'
                             'gamma = resolution_scale*sqrt(density)（零新增超参）')
    parser.add_argument('--merge-cpm-iter', action='store_true', default=False,
                        help='自适应合并判据改用「迭代式 CPM 凝聚」：反复合并 '
                             'w_ab/(|a||b|) 最大且 > gamma 的对，规模动态更新，'
                             '因此不需要 merge_ratio 配额（2026-10-30 8 数据集离线扫描：'
                             'gamma_mult=2 的平均 NMI/F1 同时优于所有固定 merge_ratio）')
    parser.add_argument('--merge-gamma-mult', type=float, default=2.0,
                        help='--merge-cpm-iter 的 gamma 倍数：'
                             'gamma = gamma_mult*resolution_scale*sqrt(density)（默认 2.0）')
    parser.add_argument('--merge-dump-dir', type=str, default='',
                        help='非空时导出「合并前划分 + 全部候选对」到该目录，'
                             '供 merge_adaptive_scan.py 离线穷举自适应判据'
                             '（不重跑流水线即可扫遍所有阈值）')

    parser.add_argument('--resolution-scales', type=str, default='auto',
                        help='逗号分隔的 resolution_scale 网格；"auto"=结构化自适应'
                             '（get_adaptive_resolution_scale），数值=固定超参，如 0.8,0.9,1.0')
    parser.add_argument('--rs-core-ref', type=float, default=None,
                        help='敏感性分析开关：临时覆盖模块级 RS_CORE_REF'
                             '（frac10core 的饱和锚点，默认 0.05）。'
                             'None=不覆盖，行为与原来逐位一致')
    parser.add_argument('--rs-core-k', type=int, default=None,
                        help='敏感性分析开关：临时覆盖模块级 RS_CORE_K'
                             '（core 存在性判据的 k，默认 10，即 10-core）。'
                             'None=不覆盖，行为与原来逐位一致')
    parser.add_argument('--rs-shape', type=str, default='sqrt',
                        choices=['sqrt', 'linear', 'log'],
                        help='Eq.3 饱和映射形状消融开关：g(s) 取 sqrt(默认,'
                             '发布配置)/linear(s)/log(log2(1+s))。三形状端点'
                             '同值 g(0)=0,g(1)=1，仅内点 s∈(0,1) 的图受影响；'
                             'sqrt 档行为与原来逐位一致')

    parser.add_argument('--no-sgm', action='store_true', default=False,
                        help='消融：禁用后处理 Stage B（选择性巨社区重建）')
    parser.add_argument('--se-cold-start', action='store_true', default=False,
                        help='warm-vs-cold 对照实验：Stage 1 SE 博弈从 singleton 冷启动'
                             '（CoDeSEG 式启动），其余不变；默认 False=warm start')

    parser.add_argument('--no-mb', action='store_true', default=False,
                        help='消融：禁用后处理 Stage A（互最优邻居合并）')
    parser.add_argument('--enable-mb', action='store_true', default=False,
                        help='恢复后处理 Stage A（2026-11-13 起默认关闭，见 ENABLE_MB 注释：'
                             'MB 在 6/8 图上用 NMI 换 F1；开启可换回 lj1/az2 的 NMI 与套件 F1）')
    parser.add_argument('--no-se-game', action='store_true', default=False,
                        help='消融：禁用 Stage 1 全局 SE 博弈（直接用块内 CPM 划分）。'
                             '全量消融实测平均为净负收益（详见私有变更日志）')
    parser.add_argument('--no-merge', action='store_true', default=False,
                        help='消融：禁用社区级成对合并 merge_communities_by_se。'
                             '2026-10-31 实测：8 数据集平均 NMI +0.07pt / F1 -0.11pt'
                             '（噪声量级）→ 与 SE 博弈功能冗余')
    parser.add_argument('--no-cpm-cleanup', action='store_true', default=False,
                        help='消融：禁用后处理 Stage D（结构自适应小社区回收，CPM 增益判据）')
    parser.add_argument('--fuse-outer', type=int, default=0,
                        help='融合模式：Stage 1（节点级 SE 迁移）与 Stage D（社区级 CPM '
                             'union）交替迭代的最大轮数，收敛到联合不动点即提前退出。'
                             '0=关闭（默认，保持 10.25 基线行为）。无新增数值超参：'
                             'tau / gamma / MIN_COMM_SIZE 全部沿用各自原有公式。')
    parser.add_argument('--sgm-t-mode', type=str, default='abs',
                        choices=['rel', 'abs'],
                        help="Stage B 碎片数门槛：'abs'=绝对阈值 3000（默认，全量回归"
                             "验证的基线行为）；'rel'=max(1000, 0.6%%·N) 随图规模"
                             "相对化（全量回归净负收益，仅留作消融）")
    parser.add_argument('--sgm-gate', type=str, default='struct',
                        choices=['census', 'struct'],
                        help="Stage B 重建门控：'census'=旧三常数（碎片≥3000 且 "
                             "占比∈[5%%,20%%]，由评测集 GT 尺寸普查标定）；"
                             "'struct'=零 GT 标定派生门控——碎片 ≥ √(2m)（复用 "
                             "Eq.8/SE 停机的度尺度派生量，无新常数）且宏外边占比 "
                             "≤ 0.15（Stage-3b 粘合判据同一物理量反向使用）。"
                             "8 图实测逐图决策一致（2026-11-11）")
    parser.add_argument('--sgm-coh-max', type=float, default=None,
                        help='[Q8 消融] 覆盖 SGM_COH_MAX（S2 外边占比上限）；'
                             'None=默认 0.15；1.0=禁用 S2（S1-only 档）')
    parser.add_argument('--sgm-frag-t', type=int, default=None,
                        help='[Q8 消融] 覆盖 S1 碎片数门槛（默认 ⌈√(2m)⌉）；'
                             '1=禁用 S1（S2-only 档）')
    parser.add_argument('--sgm2', action='store_true', default=False,
                        help='启用后处理 Stage C：迭代第二巨社区（重叠附加）')
    parser.add_argument('--hybrid-giants', type=str, default='0',
                        help='H 模式【归因诊断专用，禁止作最终报告值】：'
                             'NMI 取划分口径、F1 取"划分+前 K 个合格宏"重叠口径——'
                             '两指标来自两份不同预测输出，拼报=挑选结果。'
                             '仅用于量化 F1 缺口中"输出形态错配"的份额；'
                             '默认 0 关闭。正式报告若要走重叠口径，用 '
                             '--overlap-giants（单一重叠输出，两指标自洽）')
    parser.add_argument('--hybrid-min-frag', type=int, default=0,
                        help='H 模式附加宏的碎片数下限（0=沿用 1000；调小'
                             '可让中等规模宏进入重叠列表）')
    parser.add_argument('--hybrid-frac-min', type=float, default=None,
                        help='H 模式附加宏的占比下限（默认沿用 SGM_FRAC_MIN=0.05）。'
                             '只作用于宏评估，不改 Stage B 聚类判据——区别于 --sgm-frac-min')
    parser.add_argument('--no-mb-fallback', action='store_true', default=False,
                        help='v2 消融：禁用 MB 密度守卫触发时的小图贪心回退（回退 s8 行为）')
    parser.add_argument('--no-sg-split', action='store_true', default=False,
                        help='v2 消融：禁用 Stage F（过度合并社区定点分裂，CPM + 外边占比守卫）')

    parser.add_argument('--sg-split-max-parts', type=float, default=None,
                        help='[已删除，忽略] 防碎守卫；实测 0.30 为负收益'
                             '（切碎粘合体正是该 Stage 的收益来源）')
    parser.add_argument('--sg-split-min-frac', type=float, default=None,
                        help='[已删除，忽略] 规模阈值系数；现为 size_thr=ceil(sqrt(2E))')
    parser.add_argument('--sg-split-rel-k', type=float, default=None,
                        help='[已删除，忽略] v12 相对尺寸规则；实测与 cut 守卫职责重复')
    parser.add_argument('--sg-split-rel-min-abs', type=int, default=None,
                        help='[已删除，忽略] v12 相对规则的规模下界')
    parser.add_argument('--mb-fb-score', type=float, default=None,
                        help='[已删除，忽略] 旧 MB 小图回退的连接强度阈值；'
                             '现回退只用 ΔL<0 判据，无阈值')
    parser.add_argument('--mb-fb-max-frac', type=float, default=None,
                        help='[已删除，忽略] 旧 MB 小图回退的合并配额；现无配额')
    parser.add_argument('--alpha-max', type=float, default=None,
                        help='覆盖 cn_base_alpha 自适应上界（默认 2.0；仍为全数据集统一公式）')
    parser.add_argument('--num-processes', type=int, default=None,
                        help='并行进程数（默认 min(4, CPU核数)）；仅影响速度，不影响结果')
    parser.add_argument('--min-comm-size', type=int, default=None,
                        help='诊断：覆盖最小社区规模（默认用 get_adaptive_params 统一公式）')
    parser.add_argument('--block-size', type=int, default=None,
                        help='诊断：覆盖块大小（默认用 get_adaptive_params 统一公式）')
    parser.add_argument('--merge-gamma', type=float, default=0.30,
                        help='密度阈值合并 gamma（仅离线实验用，流水线默认不启用）')
    parser.add_argument('--merge-theta', type=float, default=0.0,
                        help='相对密度合并 theta（0=禁用；>0 启用，如 0.6）')
    parser.add_argument('--stage-eval', action='store_true', default=False,
                        help='分阶段评估：CPM后/SE博弈后/合并后/清理后各输出一次 NMI/F1')
    parser.add_argument('--save-pred', action='store_true', default=False,
                        help='保存最终预测划分到 preds/<dataset>_pred.txt（每行一个社区）')
    parser.add_argument('--se-max-iter', type=int, default=None,
                        help='[已废弃，忽略] 迭代次数现由停止判据自动决定'
                             '（#trans < N/sqrt(2E)）；保留该参数仅为兼容历史命令')
    parser.add_argument('--se-stop-mode', type=str, default=None,
                        choices=['fixed', 'trans_frac'],
                        help="SE 停止判据：fixed=仅精确不动点(#trans==0)；"
                             "trans_frac=#trans < N/sqrt(2E)（默认，零新增常数）")
    parser.add_argument('--se-cut-convention', type=str, default='legacy',
                        choices=['legacy', 'codeseg'],
                        help="SE 博弈 comm_cut 初始化约定：legacy=保持旧实现数值"
                             "（默认，结果可复现）；codeseg=对齐 CoDeSEG C++ 原版"
                             "（cut 为对外跨边权总和，不 /2），需 A/B 验证是否提升 F1")
    parser.add_argument('--no-lpa-candidate', action='store_true', default=False,
                        help='v2c 消融：禁用 Stage G（重尾缺失时的 Max-Mod LPA 候选通道），'
                             '回退到纯流水线输出')
    parser.add_argument('--lpa-runs', type=int, default=None,
                        help='Stage G：LPA 重复次数（模块度择优候选池大小，默认 3）')
    parser.add_argument('--lpa-max-frac', type=float, default=None,
                        help='[已删除，忽略] 判据 (3) 现仅作诊断打印，不再参与判定')
    parser.add_argument('--lpa-min-n', type=int, default=None,
                        help='Stage G：触发门槛 (1)——图规模下限（默认 200000）')
    parser.add_argument('--lpa-mode', type=str, default=None,
                        choices=['scratch', 'refine', 'both'],
                        help='v2c：Stage G 候选池构成（scratch=全部从头 / refine=全部'
                             '从当前划分 / both=各半，默认 both）。v2b 实际等价于 refine')
    parser.add_argument('--lpa-min-mod-gain', type=float, default=None,
                        help='[已删除，忽略] 增益判据现为 ΔQ > 0（原门槛从未绑定）')
    parser.add_argument('--lpa-min-mod-gain-rel', type=float, default=None,
                        help='[已删除，忽略] 相对门槛被绝对门槛支配（mod ≤ 1）')
    parser.add_argument('--lpa-collapse-frac', type=float, default=None,
                        help='[已删除，忽略] 塌缩防护已移除（候选池不再按规模预筛）')
    parser.add_argument('--no-lpa-shape-guard', action='store_true', default=False,
                        help='v2c 消融：禁用 Stage G 形状守卫（要求候选更重尾 + K 不塌缩）')
    parser.add_argument('--lpa-outer-iter', type=int, default=None,
                        help='[已删除，忽略] LPA 内部轮数已硬编码为 5')
    parser.add_argument('--lpa-inner-iter', type=int, default=None,
                        help='[已删除，忽略] LPA 内部扫描上限已硬编码为 20')
    parser.add_argument('--lpa-max-growth', type=float, default=None,
                        help='Stage G 形状守卫：候选最大社区相对 cur_max 的膨胀上限'
                             '（默认 6.0）。分母必须是 cur_max——纯规模判据无法区分'
                             'amazon2 的合法巨社区与 youtube 的伪巨块')
    parser.add_argument('--lpa-max-comm-mult', type=float, default=None,
                        help='[已废弃，忽略] 曾试过以 sqrt(2E) 为分母，实测在 amazon2 上误杀')
    parser.add_argument('--lpa-require-tail-gap', action='store_true', default=False,
                        help='[已删除，忽略] 判据 (3) 现仅作诊断打印，不再参与判定')
    parser.add_argument('--no-orphan-merge', action='store_true', default=False,
                        help='v2c2 消融：禁用 Stage H（孤儿微社区宿主归并）')
    parser.add_argument('--orphan-score', type=str, default=None,
                        choices=['rel', 'abs'],
                        help="v2c2：Stage H 宿主打分。'rel'=w_ab/|b|（默认，倾向"
                             "小而密的宿主）；'abs'=w_ab（倾向大宿主）")
    parser.add_argument('--orphan-max-size', type=int, default=None,
                        help='v2c2：Stage H 孤儿规模上限（默认 2）')
    parser.add_argument('--orphan-host-frac', type=float, default=None,
                        help='v2c2：Stage H 宿主规模上限（占 N 的比例，默认 0.005）')
    parser.add_argument('--mb-gamma', type=float, default=None,
                        help='[已删除，忽略] 旧 Stage A 连接强度下限 w/sqrt(|A||B|)'
                             '（历史值 0.26，扫描标定，属拟合常数；判据已改为'
                             '地图方程 ΔL<0，该参数不再读取）')
    parser.add_argument('--mb-gamma-mode', type=str, default=None,
                        help='[已删除，忽略] 旧 Stage A 判据模式开关'
                             '（gamma/dual/pure/theory/map）。现只有一种判据：'
                             '地图方程 ΔL<0，无自由常数。')
    parser.add_argument('--mb-deltaq-rho', type=float, default=None,
                        help='[已删除，忽略] 旧 ΔQ 判据 R_AB 下限；'
                             'ΔQ 判据未采用（无分辨率极限约束，见代码注释）')
    parser.add_argument('--orphan-trigger-frac', type=float, default=None,
                        help='v2c2：Stage H 触发门槛（sz<=2 社区占比，默认 0.40）')
    parser.add_argument('--tag', type=str, default='run',
                        help='输出 CSV 的标签（文件名前缀）')
    parser.add_argument('--out-dir', type=str, default='E:/HIDC-UD/results',
                        help='结果输出目录')
    args = parser.parse_args()

    if args.merge_dump_dir:
        MERGE_DUMP_DIR = args.merge_dump_dir
        os.makedirs(MERGE_DUMP_DIR, exist_ok=True)
        print(f"[MergeDump] 候选导出已启用 → {MERGE_DUMP_DIR}")

    dataset_list = [d.strip() for d in args.datasets.split(',') if d.strip()]

    resolution_scales = ['auto' if x.strip() == 'auto' else float(x)
                         for x in args.resolution_scales.split(',') if x.strip()]

    default_methods = {
        'facebook': 'metis', 'amazon1': 'metis', 'lj1': 'metis', 'dblp1': 'metis',
        'amazon2': 'metis', 'dblp2': 'metis', 'lj2': 'metis',
        'youtube': 'metis',
        'lj3': 'metis',

        **{k: 'metis' for k in DATASET_CONFIGS if k.startswith('lfr_')},
    }

    SGM_T_MODE = args.sgm_t_mode
    SGM_GATE = args.sgm_gate
    if args.sgm_coh_max is not None:
        SGM_COH_MAX = float(args.sgm_coh_max)
        print(f"[Pipeline] SGM_COH_MAX override -> {SGM_COH_MAX}")
    if args.sgm_frag_t is not None:
        SGM_FRAG_T_OVERRIDE = int(args.sgm_frag_t)
        print(f"[Pipeline] S1 frag_t override -> {SGM_FRAG_T_OVERRIDE}")
    if SGM_GATE == 'struct':
        print(f"[Pipeline] SGM_GATE=struct（零 GT 标定门控：frag≥⌈√(2m)⌉, "
              f"ext_frac≤{SGM_COH_MAX}）")
    if args.no_sgm:
        ENABLE_SGM = False
    _hg = str(args.hybrid_giants).strip().lower()
    if _hg not in ('', '0', 'none'):
        if _hg == 'auto':
            HYBRID_AUTO = True
            print(f"[Pipeline] HYBRID=auto（H 模式·先验门控：仅 "
                  f"{sorted(_HYBRID_OVERLAP_PRIOR)} 家族启用重叠 F1 口径，"
                  f"NMI 恒为划分口径；不读 GT，判据=network_type 元信息）")
        else:
            HYBRID_GIANTS_K = int(_hg)
            print(f"[Pipeline] HYBRID_GIANTS_K={HYBRID_GIANTS_K}（H 模式·强制档："
                  f"全部数据集启用重叠 F1 口径，NMI 仍为划分口径）")
        if args.hybrid_min_frag and int(args.hybrid_min_frag) > 0:
            HYBRID_MIN_FRAG = int(args.hybrid_min_frag)
        if args.hybrid_frac_min is not None:
            HYBRID_FRAC_MIN = max(0.0, float(args.hybrid_frac_min))
        print(f"[Pipeline]   宏守卫: min_frag={HYBRID_MIN_FRAG}, "
              f"frac_min={'SGM(0.05)' if HYBRID_FRAC_MIN is None else HYBRID_FRAC_MIN}"
              f"（H 模式独立，不影响 Stage B）")
    if args.se_cold_start:
        SE_COLD_START = True

    if args.no_mb:
        ENABLE_MB = False
    if args.enable_mb:
        ENABLE_MB = True
    if args.no_cpm_cleanup:
        ENABLE_CPM_CLEANUP = False
    if int(args.fuse_outer or 0) > 0:
        FUSE_OUTER = int(args.fuse_outer)
        print(f"[Pipeline] FUSE_OUTER={FUSE_OUTER} "
              f"(Stage1 SE <-> StageD CPM 交替到联合不动点)")
    if args.no_mb_fallback:
        ENABLE_MB_FALLBACK = False
    if args.no_sg_split:
        ENABLE_SG_SPLIT = False

    if args.alpha_max is not None:
        CN_ALPHA_MAX = float(args.alpha_max)
    if args.num_processes is not None:
        NUM_PROCESSES = max(1, int(args.num_processes))
    if args.min_comm_size is not None:
        MIN_COMM_SIZE_OVERRIDE = int(args.min_comm_size)
    if args.block_size is not None:
        BLOCK_SIZE_OVERRIDE = int(args.block_size)
    if args.rs_core_ref is not None:

        RS_CORE_REF = float(args.rs_core_ref)
        print(f"[Pipeline] RS_CORE_REF override = {RS_CORE_REF}")
    if args.rs_core_k is not None:

        RS_CORE_K = int(args.rs_core_k)
        print(f"[Pipeline] RS_CORE_K override = {RS_CORE_K}")
    if getattr(args, "rs_shape", None) and args.rs_shape != "sqrt":

        RS_SHAPE = str(args.rs_shape)
        print(f"[Pipeline] RS_SHAPE override = {RS_SHAPE}")
    SE_CUT_CONVENTION = args.se_cut_convention
    if args.se_stop_mode:
        SE_STOP_MODE = args.se_stop_mode
    if args.no_lpa_candidate:
        ENABLE_LPA_CANDIDATE = False
    if args.lpa_runs is not None:
        LPA_CAND_RUNS = max(1, int(args.lpa_runs))
    if args.lpa_mode:
        LPA_CAND_MODE = str(args.lpa_mode).lower()
    if args.no_lpa_shape_guard:
        LPA_CAND_SHAPE_GUARD = False
    if args.lpa_max_growth is not None:
        LPA_CAND_MAX_GROWTH = float(args.lpa_max_growth)
    if args.lpa_min_n is not None:
        LPA_CAND_MIN_N = int(args.lpa_min_n)
    if args.no_orphan_merge:
        ENABLE_ORPHAN_MERGE = False
    if args.orphan_score:
        ORPHAN_SCORE = str(args.orphan_score).lower()
    if args.orphan_max_size is not None:
        ORPHAN_MAX_SIZE = int(args.orphan_max_size)
    if args.orphan_host_frac is not None:
        ORPHAN_HOST_MAX_FRAC = float(args.orphan_host_frac)
    if args.orphan_trigger_frac is not None:
        ORPHAN_TRIGGER_FRAC = float(args.orphan_trigger_frac)
    if args.mb_gamma is not None:
        print("[Pipeline] 注意：--mb-gamma 已失效（判据改为地图方程 ΔL<0，"
              "无连接强度阈值），本次运行忽略该参数。")
    if args.mb_gamma_mode is not None:
        print("[Pipeline] 注意：--mb-gamma-mode 已失效（Stage A 只有一种判据："
              f"ΔL<0；忽略 mode={args.mb_gamma_mode}）。")
    if args.mb_deltaq_rho is not None:
        print("[Pipeline] 注意：--mb-deltaq-rho 已失效（ΔQ 判据未采用），忽略。")
    print(f"[Pipeline] ENABLE_MB={ENABLE_MB}, ENABLE_SGM={ENABLE_SGM}, "
          f"SGM_T_MODE={SGM_T_MODE}, "
          f"CN_ALPHA_MAX={CN_ALPHA_MAX}, NUM_PROCESSES={NUM_PROCESSES}, "
          f"MIN_COMM_SIZE_OVERRIDE={MIN_COMM_SIZE_OVERRIDE}, "
          f"BLOCK_SIZE_OVERRIDE={BLOCK_SIZE_OVERRIDE}, "
          f"SE_CUT={SE_CUT_CONVENTION}, "

          f"STAGE_G(enabled={ENABLE_LPA_CANDIDATE}, min_n={LPA_CAND_MIN_N}, "
          f"max_deg={LPA_CAND_MAX_DEG}, runs={LPA_CAND_RUNS}, mode={LPA_CAND_MODE}), "
          f"MB_CRITERION='mutual-best + significance ΔS>0'（Traag 2013，零自由常数）")

    experiments = []

    for rs in resolution_scales:

            for dataset in dataset_list:
                experiments.append((dataset, rs))

    results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for dataset_name, resolution_scale in experiments:
        config = DATASET_CONFIGS.get(dataset_name)
        if config is None:
            print(f"ERROR: Unknown dataset {dataset_name}")
            continue

        EDGE_FILE_PATH = config["edge_path"]
        COMMUNITY_FILE_PATH = config["community_path"]
        network_type = config["network_type"]
        method = default_methods.get(dataset_name, 'bfs')

        print("\n" + "=" * 70)
        print(f"  DATASET: {dataset_name}  METHOD: {method}  —  {config['description']}")
        print(f"  network_type={network_type}, "
              f"resolution_scale={resolution_scale}, se_stop_mode={SE_STOP_MODE}")
        print(f"  Edge file:      {EDGE_FILE_PATH}")
        print(f"  Community file: {COMMUNITY_FILE_PATH}")
        print("=" * 70 + "\n")

        if not os.path.exists(EDGE_FILE_PATH):
            print(f"ERROR: Edge file not found: {EDGE_FILE_PATH}")
            continue

        if not os.path.exists(COMMUNITY_FILE_PATH):
            print(f"WARNING: Community file not found: {COMMUNITY_FILE_PATH}")

        start = time.time()
        rs_out = {}
        try:
            nmi_val, f1_val = execute_ASNCD_pipeline_unsupervised(
                EDGE_FILE_PATH, COMMUNITY_FILE_PATH, dataset_name, network_type,
                partition_method=method,
                resolution_scale=resolution_scale,

                stage_eval=args.stage_eval,
                save_pred=args.save_pred,

                merge_score_mult=(args.merge_score_mult if args.merge_score_mult > 0 else None),
                merge_cap=(args.merge_cap if args.merge_cap > 0 else None),
                merge_cpm=args.merge_cpm,
                merge_cpm_iter=args.merge_cpm_iter,
                merge_gamma_mult=args.merge_gamma_mult,
                enable_se_game=(not args.no_se_game),
                enable_merge=(not args.no_merge),
                rs_out=rs_out
            )
            elapsed = time.time() - start
            results.append({
                'dataset': dataset_name, 'method': method,
                'network_type': network_type,

                'merge_score_mult': (args.merge_score_mult if args.merge_score_mult > 0 else ''),
                'resolution_scale': rs_out.get('resolution_scale', resolution_scale),
                'rs_mode': 'auto' if resolution_scale == 'auto' else 'fixed',
                'frac10core': rs_out.get('frac10core', ''),
                'lcc_frac': rs_out.get('lcc_frac', ''),

                'nmi': nmi_val, 'f1': f1_val,
                'time_sec': elapsed, 'status': 'OK'
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'dataset': dataset_name, 'method': method,
                'network_type': network_type,

                'resolution_scale': rs_out.get('resolution_scale', resolution_scale),
                'rs_mode': 'auto' if resolution_scale == 'auto' else 'fixed',
                'frac10core': rs_out.get('frac10core', ''),
                'lcc_frac': rs_out.get('lcc_frac', ''),

                'nmi': None, 'f1': None,
                'time_sec': elapsed, 'status': 'ERROR'
            })

        if results:
            os.makedirs(args.out_dir, exist_ok=True)
            partial_path = os.path.join(args.out_dir, f"s8_{args.tag}_{timestamp}.csv")
            pd.DataFrame(results).to_csv(partial_path, index=False)

    if results:
        os.makedirs(args.out_dir, exist_ok=True)
        summary_path = os.path.join(args.out_dir, f"s8_{args.tag}_{timestamp}.csv")
        df = pd.DataFrame(results)
        df.to_csv(summary_path, index=False)
        print(f"\n[SUMMARY] saved to {summary_path}")
        print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  ALL DATASETS COMPLETE")
    print("=" * 70)
