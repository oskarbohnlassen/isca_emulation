from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree
from torch.utils.data import IterableDataset, get_worker_info

from isca_emulation_v2.data.utils import dump_yaml_mapping, load_isca_result_data


@dataclass(frozen=True)
class TriangularMesh:
    vertices: np.ndarray
    faces: np.ndarray


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64).copy()
    for idx, face in enumerate(faces):
        tri = vertices[face]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if float(np.dot(normal, tri.mean(axis=0))) < 0.0:
            faces[idx] = np.asarray([face[0], face[2], face[1]], dtype=np.int64)
    return faces


def _build_icosahedron() -> TriangularMesh:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    vertices = _normalize_vectors(vertices)
    faces = ConvexHull(vertices).simplices
    faces = _orient_faces_outward(vertices, faces)
    return TriangularMesh(vertices=vertices, faces=faces)


def _subdivide_mesh(mesh: TriangularMesh) -> TriangularMesh:
    vertices = [vertex.copy() for vertex in np.asarray(mesh.vertices, dtype=np.float64)]
    midpoint_cache: dict[tuple[int, int], int] = {}
    new_faces: list[list[int]] = []

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        mid = _normalize_vectors(((vertices[i] + vertices[j]) * 0.5).reshape(1, 3))[0]
        idx = len(vertices)
        vertices.append(mid)
        midpoint_cache[key] = idx
        return idx

    for i, j, k in np.asarray(mesh.faces, dtype=np.int64):
        a = midpoint(int(i), int(j))
        b = midpoint(int(j), int(k))
        c = midpoint(int(k), int(i))
        new_faces.extend(
            [
                [int(i), a, c],
                [a, int(j), b],
                [c, b, int(k)],
                [a, b, c],
            ]
        )

    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = _orient_faces_outward(vertices_np, np.asarray(new_faces, dtype=np.int64))
    return TriangularMesh(vertices=vertices_np, faces=faces_np)


def build_mesh_hierarchy(mesh_splits: int) -> list[TriangularMesh]:
    if mesh_splits < 0:
        raise ValueError("mesh_splits must be >= 0.")
    meshes = [_build_icosahedron()]
    for _ in range(mesh_splits):
        meshes.append(_subdivide_mesh(meshes[-1]))
    return meshes


def _faces_to_bidirected_edges(faces: np.ndarray) -> np.ndarray:
    undirected_edges: set[tuple[int, int]] = set()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge = (u, v) if u < v else (v, u)
            undirected_edges.add(edge)

    directed_edges: list[tuple[int, int]] = []
    for u, v in sorted(undirected_edges):
        directed_edges.append((u, v))
        directed_edges.append((v, u))
    return np.asarray(directed_edges, dtype=np.int64).T


def _max_edge_distance(vertices: np.ndarray, faces: np.ndarray) -> float:
    undirected_edges = _faces_to_bidirected_edges(faces)[:, ::2]
    src = vertices[undirected_edges[0]]
    dst = vertices[undirected_edges[1]]
    return float(np.linalg.norm(dst - src, axis=-1).max())


def _xyz_to_latlon(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = np.clip(vertices[:, 2], -1.0, 1.0)
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)
    return lat, lon


def _node_features_from_xyz(vertices: np.ndarray) -> np.ndarray:
    lat, lon = _xyz_to_latlon(vertices)
    return np.concatenate(
        [
            vertices.astype(np.float32),
            np.sin(lat).reshape(-1, 1).astype(np.float32),
            np.cos(lat).reshape(-1, 1).astype(np.float32),
            np.sin(lon).reshape(-1, 1).astype(np.float32),
            np.cos(lon).reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    )


def _edge_features_from_xyz(sender_xyz: np.ndarray, receiver_xyz: np.ndarray) -> np.ndarray:
    delta = receiver_xyz - sender_xyz
    chord = np.linalg.norm(delta, axis=-1, keepdims=True)
    dot = np.sum(sender_xyz * receiver_xyz, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    great_circle = np.arccos(dot)
    return np.concatenate([delta.astype(np.float32), chord.astype(np.float32), great_circle.astype(np.float32)], axis=1)


def _safe_normalize(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = vectors / np.clip(norms, 1e-12, None)
    if np.all(norms > 1e-12):
        return safe
    return np.where(norms > 1e-12, safe, fallback)


def _edge_features_receiver_local(sender_xyz: np.ndarray, receiver_xyz: np.ndarray) -> np.ndarray:
    """Encode sender positions in a receiver-local 3D frame."""
    up = _normalize_vectors(receiver_xyz)
    reference = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64), (up.shape[0], 1))
    east = np.cross(reference, up)

    degenerate = np.linalg.norm(east, axis=-1, keepdims=True) <= 1e-12
    if np.any(degenerate):
        alt_reference = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64), (up.shape[0], 1))
        east = np.where(degenerate, np.cross(alt_reference, up), east)

    east = _safe_normalize(east, fallback=np.zeros_like(east))
    north = _safe_normalize(np.cross(up, east), fallback=np.zeros_like(up))

    relative = sender_xyz - receiver_xyz
    relative_local = np.stack(
        [
            np.sum(relative * up, axis=-1),
            np.sum(relative * east, axis=-1),
            np.sum(relative * north, axis=-1),
        ],
        axis=-1,
    )
    edge_distance = np.linalg.norm(relative_local, axis=-1, keepdims=True)
    normalization = float(np.clip(edge_distance.max(initial=0.0), 1e-12, None))
    return np.concatenate(
        [
            (edge_distance / normalization).astype(np.float32),
            (relative_local / normalization).astype(np.float32),
        ],
        axis=-1,
    )


def _build_edge_features(
    sender_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    *,
    edge_feature_mode: str,
) -> tuple[np.ndarray, list[str]]:
    edge_feature_mode = str(edge_feature_mode).lower()
    if edge_feature_mode == "global_xyz":
        return _edge_features_from_xyz(sender_xyz, receiver_xyz), [
            "delta_x",
            "delta_y",
            "delta_z",
            "chord_distance",
            "great_circle_angle",
        ]
    if edge_feature_mode == "receiver_local":
        return _edge_features_receiver_local(sender_xyz, receiver_xyz), [
            "local_distance",
            "local_receiver_x",
            "local_receiver_y",
            "local_receiver_z",
        ]
    raise ValueError("edge_feature_mode must be one of ['global_xyz', 'receiver_local'].")


def _grid_lat_lon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64).reshape(-1))
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=np.float64).reshape(-1))
    lat_grid, lon_grid = np.meshgrid(lat_rad, lon_rad, indexing="ij")
    cos_lat = np.cos(lat_grid)
    xyz = np.stack(
        [
            cos_lat * np.cos(lon_grid),
            cos_lat * np.sin(lon_grid),
            np.sin(lat_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
    return xyz.astype(np.float64), lat_grid.reshape(-1), lon_grid.reshape(-1)


def _point_in_spherical_triangle(point_xyz: np.ndarray, triangle_xyz: np.ndarray, eps: float = 1e-9) -> bool:
    a, b, c = triangle_xyz
    tests = np.asarray(
        [
            np.dot(np.cross(a, b), point_xyz),
            np.dot(np.cross(b, c), point_xyz),
            np.dot(np.cross(c, a), point_xyz),
        ],
        dtype=np.float64,
    )
    return bool(np.all(tests >= -eps) or np.all(tests <= eps))


def _find_containing_face_indices(
    grid_xyz: np.ndarray,
    face_vertices: np.ndarray,
    *,
    candidate_face_k: int,
) -> np.ndarray:
    face_centroids = _normalize_vectors(face_vertices.mean(axis=1))
    tree = cKDTree(face_centroids)
    k = min(int(candidate_face_k), int(face_vertices.shape[0]))
    candidate_faces = tree.query(grid_xyz, k=k)[1]
    if k == 1:
        candidate_faces = candidate_faces.reshape(-1, 1)

    selected = np.empty((grid_xyz.shape[0],), dtype=np.int64)
    for idx, point_xyz in enumerate(grid_xyz):
        found_face = None
        for face_idx in np.asarray(candidate_faces[idx], dtype=np.int64).reshape(-1):
            if _point_in_spherical_triangle(point_xyz, face_vertices[int(face_idx)]):
                found_face = int(face_idx)
                break
        if found_face is None:
            for face_idx, tri in enumerate(face_vertices):
                if _point_in_spherical_triangle(point_xyz, tri):
                    found_face = int(face_idx)
                    break
        if found_face is None:
            found_face = int(np.asarray(candidate_faces[idx]).reshape(-1)[0])
        selected[idx] = found_face
    return selected


def build_mesh_gnn_static_graph(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    mesh_splits: int,
    radius_query_factor: float = 0.6,
    candidate_face_k: int = 16,
    edge_feature_mode: str = "global_xyz",
) -> dict[str, Any]:
    if radius_query_factor <= 0.0:
        raise ValueError("radius_query_factor must be > 0.")
    if candidate_face_k < 1:
        raise ValueError("candidate_face_k must be >= 1.")

    mesh_hierarchy = build_mesh_hierarchy(mesh_splits)
    finest_mesh = mesh_hierarchy[-1]
    grid_xyz, grid_lat_rad, grid_lon_rad = _grid_lat_lon_to_xyz(lat, lon)
    finest_vertices = finest_mesh.vertices
    finest_faces = finest_mesh.faces

    level_vertex_counts = [int(mesh.vertices.shape[0]) for mesh in mesh_hierarchy]
    vertex_first_level = np.zeros((finest_vertices.shape[0],), dtype=np.int64)
    prev_count = 0
    for level_idx, count in enumerate(level_vertex_counts):
        vertex_first_level[prev_count:count] = level_idx
        prev_count = count

    multimesh_faces = np.concatenate([mesh.faces for mesh in mesh_hierarchy], axis=0)
    mesh_face_level = np.concatenate(
        [
            np.full((mesh.faces.shape[0],), fill_value=level_idx, dtype=np.int64)
            for level_idx, mesh in enumerate(mesh_hierarchy)
        ],
        axis=0,
    )
    mesh_edge_index_np = _faces_to_bidirected_edges(multimesh_faces)

    edge_radius = radius_query_factor * _max_edge_distance(finest_vertices, finest_faces)
    mesh_tree = cKDTree(finest_vertices)
    grid_to_mesh_lists = mesh_tree.query_ball_point(grid_xyz, r=edge_radius)

    g2m_src: list[int] = []
    g2m_dst: list[int] = []
    for grid_idx, neighbours in enumerate(grid_to_mesh_lists):
        if len(neighbours) == 0:
            nearest = int(mesh_tree.query(grid_xyz[grid_idx], k=1)[1])
            neighbours = [nearest]
        for mesh_idx in neighbours:
            g2m_src.append(int(grid_idx))
            g2m_dst.append(int(mesh_idx))
    g2m_edge_index_np = np.asarray([g2m_src, g2m_dst], dtype=np.int64)

    containing_faces = _find_containing_face_indices(
        grid_xyz,
        finest_vertices[finest_faces],
        candidate_face_k=candidate_face_k,
    )
    m2g_src: list[int] = []
    m2g_dst: list[int] = []
    for grid_idx, face_idx in enumerate(containing_faces):
        tri = finest_faces[int(face_idx)]
        for mesh_idx in tri:
            m2g_src.append(int(mesh_idx))
            m2g_dst.append(int(grid_idx))
    m2g_edge_index_np = np.asarray([m2g_src, m2g_dst], dtype=np.int64)

    grid_node_features_np = _node_features_from_xyz(grid_xyz)
    mesh_node_features_np = _node_features_from_xyz(finest_vertices)
    mesh_edge_features_np, edge_feature_names = _build_edge_features(
        finest_vertices[mesh_edge_index_np[0]],
        finest_vertices[mesh_edge_index_np[1]],
        edge_feature_mode=edge_feature_mode,
    )
    g2m_edge_features_np, g2m_edge_feature_names = _build_edge_features(
        grid_xyz[g2m_edge_index_np[0]],
        finest_vertices[g2m_edge_index_np[1]],
        edge_feature_mode=edge_feature_mode,
    )
    m2g_edge_features_np, m2g_edge_feature_names = _build_edge_features(
        finest_vertices[m2g_edge_index_np[0]],
        grid_xyz[m2g_edge_index_np[1]],
        edge_feature_mode=edge_feature_mode,
    )
    if g2m_edge_feature_names != edge_feature_names or m2g_edge_feature_names != edge_feature_names:
        raise RuntimeError("MeshGNN edge feature schemas must match across edge sets.")

    hierarchy_payload = {
        "vertices": [torch.as_tensor(mesh.vertices, dtype=torch.float32) for mesh in mesh_hierarchy],
        "faces": [torch.as_tensor(mesh.faces, dtype=torch.long) for mesh in mesh_hierarchy],
        "num_levels": int(len(mesh_hierarchy)),
        "num_vertices_per_level": level_vertex_counts,
        "num_faces_per_level": [int(mesh.faces.shape[0]) for mesh in mesh_hierarchy],
    }

    return {
        "grid": {
            "lat_deg": torch.as_tensor(np.asarray(lat, dtype=np.float32).reshape(-1), dtype=torch.float32),
            "lon_deg": torch.as_tensor(np.asarray(lon, dtype=np.float32).reshape(-1), dtype=torch.float32),
            "node_lat_rad": torch.as_tensor(grid_lat_rad.astype(np.float32), dtype=torch.float32),
            "node_lon_rad": torch.as_tensor(grid_lon_rad.astype(np.float32), dtype=torch.float32),
            "node_xyz": torch.as_tensor(grid_xyz.astype(np.float32), dtype=torch.float32),
            "node_features": torch.as_tensor(grid_node_features_np, dtype=torch.float32),
            "num_nodes": int(grid_xyz.shape[0]),
            "shape": [int(len(lat)), int(len(lon))],
        },
        "mesh_hierarchy": hierarchy_payload,
        "mesh": {
            "node_xyz": torch.as_tensor(finest_vertices.astype(np.float32), dtype=torch.float32),
            "node_lat_rad": torch.as_tensor(_xyz_to_latlon(finest_vertices)[0].astype(np.float32), dtype=torch.float32),
            "node_lon_rad": torch.as_tensor(_xyz_to_latlon(finest_vertices)[1].astype(np.float32), dtype=torch.float32),
            "node_features": torch.as_tensor(mesh_node_features_np, dtype=torch.float32),
            "vertex_first_level": torch.as_tensor(vertex_first_level, dtype=torch.long),
            "faces": torch.as_tensor(multimesh_faces, dtype=torch.long),
            "face_level": torch.as_tensor(mesh_face_level, dtype=torch.long),
            "num_nodes": int(finest_vertices.shape[0]),
            "num_faces": int(multimesh_faces.shape[0]),
        },
        "grid2mesh": {
            "edge_index": torch.as_tensor(g2m_edge_index_np, dtype=torch.long),
            "edge_features": torch.as_tensor(g2m_edge_features_np, dtype=torch.float32),
            "num_edges": int(g2m_edge_index_np.shape[1]),
            "radius_query_factor": float(radius_query_factor),
            "radius_query": float(edge_radius),
        },
        "mesh_graph": {
            "edge_index": torch.as_tensor(mesh_edge_index_np, dtype=torch.long),
            "edge_features": torch.as_tensor(mesh_edge_features_np, dtype=torch.float32),
            "num_edges": int(mesh_edge_index_np.shape[1]),
            "bidirected": True,
        },
        "mesh2grid": {
            "edge_index": torch.as_tensor(m2g_edge_index_np, dtype=torch.long),
            "edge_features": torch.as_tensor(m2g_edge_features_np, dtype=torch.float32),
            "num_edges": int(m2g_edge_index_np.shape[1]),
            "edges_per_grid_node": 3,
        },
        "feature_schema": {
            "node_features": ["x", "y", "z", "sin_lat", "cos_lat", "sin_lon", "cos_lon"],
            "edge_features": edge_feature_names,
            "edge_feature_mode": str(edge_feature_mode),
        },
    }


def xarray_to_mesh_gnn_tensor(
    ds,
    vars: list[str],
    *,
    level_dim: str,
    lat_dim: str,
    lon_dim: str,
    time_dim: str,
    load: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    blocks: list[np.ndarray] = []
    for var_name in vars:
        da = ds[var_name].transpose(time_dim, level_dim, lat_dim, lon_dim)
        if load:
            da = da.load()
        blocks.append(np.asarray(da.to_numpy(), dtype=np.float32))

    dyn = np.concatenate(blocks, axis=1)
    t_size, f_dyn, nlat, nlon = dyn.shape
    x_dyn = np.transpose(dyn, (0, 2, 3, 1)).reshape(t_size, nlat * nlon, f_dyn)

    meta = {
        "selected_levels": np.asarray(ds[level_dim].values, dtype=np.float32).reshape(-1).tolist(),
        "dynamic_feature_order": "variable-major, then level-major",
        "num_dynamic_features": int(f_dyn),
    }
    return torch.from_numpy(np.ascontiguousarray(x_dyn)), meta


def _load_external_stats(path_like: str | Path) -> dict[str, torch.Tensor]:
    stats = torch.load(Path(path_like), map_location="cpu", weights_only=True)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)
    return {"mean": mean, "std": std.clamp_min(1e-6)}


def inverse_scale_mesh_gnn(scaled_data: np.ndarray, cfg) -> np.ndarray:
    path_to_stats = os.path.join("data", "processed", cfg.data.output.dataset_name, "stats.pt")
    stats = torch.load(path_to_stats, map_location="cpu", weights_only=False)
    mean = stats["mean"].reshape(-1).cpu().detach().numpy()
    std = stats["std"].reshape(-1).cpu().detach().numpy()
    return scaled_data * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


class SimpleDatasetMeshGNN(IterableDataset):
    """Stream contiguous timestep pairs plus MeshGNN static artifacts."""

    def __init__(
        self,
        shard_paths: list[str],
        shard_lengths: list[int],
        static_path: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        batch_size: int,
        shuffle_shards: bool = False,
        shuffle_in_shard: bool = False,
        cache_in_memory: bool = False,
    ):
        self.shard_paths = list(shard_paths)
        self.shard_lengths = [int(v) for v in shard_lengths]
        self.pairs_per_shard = [v - 1 for v in self.shard_lengths]
        self.batch_size = int(batch_size)
        self.total_pairs = int(sum(self.pairs_per_shard))
        self.total_batches = int(sum((n + self.batch_size - 1) // self.batch_size for n in self.pairs_per_shard))
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_in_shard = bool(shuffle_in_shard)
        self.cache_in_memory = bool(cache_in_memory)
        self.static = torch.load(static_path, map_location="cpu", weights_only=False)
        self._cached_shards = None

        self._mean_vec = mean.reshape(-1).to(torch.float32)
        self._std_vec = std.reshape(-1).to(torch.float32).clamp_min(1e-6)
        self.grid_static = self.static["grid"]
        self.mesh_static = self.static["mesh"]
        self.grid2mesh = self.static["grid2mesh"]
        self.mesh_graph = self.static["mesh_graph"]
        self.mesh2grid = self.static["mesh2grid"]

        if self.cache_in_memory:
            self._cached_shards = [self._prepare_shard(shard_idx) for shard_idx in range(len(self.shard_paths))]

    def _load_shard(self, shard_idx: int, *, mmap: bool) -> dict[str, torch.Tensor]:
        return torch.load(self.shard_paths[shard_idx], weights_only=False, mmap=mmap)

    def _prepare_shard(self, shard_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self._load_shard(shard_idx, mmap=False)
        x_dyn = shard["x_dyn"].to(torch.float32)
        mean = self._mean_vec.view(1, 1, -1)
        std = self._std_vec.view(1, 1, -1)
        return (x_dyn[:-1] - mean) / std, (x_dyn[1:] - mean) / std

    def _iter_shard_batches(self, shard_idx: int):
        shard = self._load_shard(shard_idx, mmap=True)
        x_dyn = shard["x_dyn"]
        num_pairs = int(x_dyn.shape[0]) - 1
        if num_pairs <= 0:
            return

        mean = self._mean_vec.view(1, 1, -1)
        std = self._std_vec.view(1, 1, -1)
        if self.shuffle_in_shard:
            sample_order = torch.randperm(num_pairs)
            for i in range(0, num_pairs, self.batch_size):
                idx = sample_order[i : i + self.batch_size]
                yield (x_dyn[idx].to(torch.float32) - mean) / std, (x_dyn[idx + 1].to(torch.float32) - mean) / std
        else:
            for i in range(0, num_pairs, self.batch_size):
                j = min(num_pairs, i + self.batch_size)
                yield (x_dyn[i:j].to(torch.float32) - mean) / std, (x_dyn[i + 1 : j + 1].to(torch.float32) - mean) / std

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            shard_idxs = list(range(len(self.shard_paths)))
        else:
            shard_idxs = list(range(worker.id, len(self.shard_paths), worker.num_workers))

        if self.shuffle_shards:
            shard_order = torch.randperm(len(shard_idxs)).tolist()
            shard_idxs = [shard_idxs[i] for i in shard_order]

        for shard_idx in shard_idxs:
            if self.cache_in_memory:
                x_state, y_state = self._cached_shards[shard_idx]
                if self.shuffle_in_shard:
                    sample_order = torch.randperm(x_state.shape[0])
                    x_state = x_state[sample_order]
                    y_state = y_state[sample_order]
                for i in range(0, x_state.shape[0], self.batch_size):
                    j = min(x_state.shape[0], i + self.batch_size)
                    yield x_state[i:j], y_state[i:j]
            else:
                yield from self._iter_shard_batches(shard_idx)


def process_mesh_gnn(
    *,
    raw_cfg: dict[str, Any],
    proc_cfg: dict[str, Any],
    out_cfg: dict[str, Any],
    split_cfg: list[float] | bool,
    split_ranges_fn: Callable[[int, list[float] | bool], dict[str, tuple[int, int]]],
    stats_path: str | None,
    processed_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    proc_params = proc_cfg["params"]

    exp_folder_name = raw_cfg["exp_folder_name"]
    file_name = raw_cfg["file_name"]
    variables = list(proc_params["vars"])
    level_dim = proc_params["level_dim"]
    lat_dim = proc_params["lat_dim"]
    lon_dim = proc_params["lon_dim"]
    time_dim = proc_params["time_dim"]
    time_chunk = int(proc_params.get("time_chunk", 240))
    time_start = int(proc_params.get("time_start", 0))
    max_timesteps = proc_params.get("max_timesteps")
    shard_pairs = int(proc_params.get("shard_pairs", time_chunk))
    mesh_splits = int(proc_params.get("mesh_splits", 2))
    radius_query_factor = float(proc_params.get("radius_query_factor", 0.6))
    candidate_face_k = int(proc_params.get("candidate_face_k", 16))
    edge_feature_mode = str(proc_params.get("edge_feature_mode", "global_xyz"))

    processed_dir.mkdir(parents=True, exist_ok=False)

    ds = load_isca_result_data(exp_folder_name=exp_folder_name, file_name=file_name)
    total_time_all = int(ds.sizes[time_dim])
    end_idx = total_time_all if max_timesteps is None else min(total_time_all, time_start + int(max_timesteps))
    ds = ds.isel({time_dim: slice(time_start, end_idx)})
    total_time = int(ds.sizes[time_dim])

    lat_vals = np.asarray(ds[lat_dim].values, dtype=np.float32).reshape(-1)
    lon_vals = np.asarray(ds[lon_dim].values, dtype=np.float32).reshape(-1)
    static_graph = build_mesh_gnn_static_graph(
        lat_vals,
        lon_vals,
        mesh_splits=mesh_splits,
        radius_query_factor=radius_query_factor,
        candidate_face_k=candidate_face_k,
        edge_feature_mode=edge_feature_mode,
    )
    static_graph["dynamic_features"] = {
        "vars": list(variables),
        "level_dim": str(level_dim),
        "lat_dim": str(lat_dim),
        "lon_dim": str(lon_dim),
        "time_dim": str(time_dim),
    }
    torch.save(static_graph, processed_dir / "mesh_gnn_static.pt")

    split_ranges = split_ranges_fn(total_time, split_cfg)
    split_names = list(split_ranges.keys())
    stats_source = "all" if "all" in split_names else "train"
    split_buffers: dict[str, torch.Tensor | None] = {name: None for name in split_names}
    split_shards: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    split_next_idx: dict[str, int] = {name: 0 for name in split_names}
    split_total_timesteps: dict[str, int] = {name: 0 for name in split_names}

    stats_sum = None
    stats_sumsq = None
    stats_count = None
    feature_shape: list[int] | None = None
    feature_meta: dict[str, Any] | None = None

    def append_split(name: str, piece: torch.Tensor) -> None:
        nonlocal stats_sum, stats_sumsq, stats_count
        split_total_timesteps[name] += int(piece.shape[0])

        if stats_path is None and name == stats_source:
            piece64 = piece.to(torch.float64)
            reduce_dims = (0, 1)
            finite = torch.isfinite(piece64)
            finite_piece = torch.where(finite, piece64, torch.zeros_like(piece64))
            chunk_sum = finite_piece.sum(dim=reduce_dims, keepdim=True)
            chunk_sumsq = (finite_piece * finite_piece).sum(dim=reduce_dims, keepdim=True)
            chunk_count = finite.to(torch.float64).sum(dim=reduce_dims, keepdim=True)
            if stats_sum is None:
                stats_sum = torch.zeros_like(chunk_sum)
                stats_sumsq = torch.zeros_like(chunk_sumsq)
                stats_count = torch.zeros_like(chunk_count)
            stats_sum += chunk_sum
            stats_sumsq += chunk_sumsq
            stats_count += chunk_count

        if split_buffers[name] is None:
            split_buffers[name] = piece
        else:
            split_buffers[name] = torch.cat([split_buffers[name], piece], dim=0)

        while split_buffers[name].shape[0] >= shard_pairs + 1:
            shard_tensor = split_buffers[name][: shard_pairs + 1].clone()
            shard_name = f"{name}_{split_next_idx[name]:03d}.pt"
            torch.save({"x_dyn": shard_tensor}, processed_dir / shard_name)
            split_shards[name].append({"path": shard_name, "shape": list(shard_tensor.shape)})
            split_next_idx[name] += 1
            split_buffers[name] = split_buffers[name][shard_pairs:]

    for start in range(0, total_time, time_chunk):
        end = min(total_time, start + time_chunk)
        ds_chunk = ds.isel({time_dim: slice(start, end)})
        x_chunk, meta = xarray_to_mesh_gnn_tensor(
            ds_chunk,
            vars=variables,
            level_dim=level_dim,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
            time_dim=time_dim,
            load=True,
        )
        if feature_shape is None:
            feature_shape = [int(v) for v in x_chunk.shape[1:]]
            feature_meta = meta

        for name, (split_start, split_end) in split_ranges.items():
            local_start = max(start, split_start)
            local_end = min(end, split_end)
            if local_end > local_start:
                chunk_start = local_start - start
                chunk_end = local_end - start
                append_split(name, x_chunk[chunk_start:chunk_end])

    for name in split_names:
        if split_buffers[name] is not None and split_buffers[name].shape[0] >= 2:
            shard_tensor = split_buffers[name].clone()
            shard_name = f"{name}_{split_next_idx[name]:03d}.pt"
            torch.save({"x_dyn": shard_tensor}, processed_dir / shard_name)
            split_shards[name].append({"path": shard_name, "shape": list(shard_tensor.shape)})

    if stats_path is not None:
        stats = _load_external_stats(stats_path)
        stats_split = "external"
    else:
        valid_count = stats_count.clamp_min(1.0)
        mean64 = stats_sum / valid_count
        var64 = stats_sumsq / valid_count - mean64 * mean64
        var64 = torch.where(stats_count > 0, var64, torch.ones_like(var64))
        mean = mean64.to(torch.float32)
        std = torch.sqrt(var64.clamp_min(1e-12)).to(torch.float32).clamp_min(1e-6)
        f_dyn = int(feature_shape[-1]) if feature_shape is not None else int(mean.shape[-1])
        stats = {"mean": mean.reshape(1, f_dyn, 1, 1), "std": std.reshape(1, f_dyn, 1, 1)}
        stats_split = stats_source

    torch.save(stats, processed_dir / "stats.pt")
    split_shapes = {name: [split_total_timesteps[name], *(feature_shape or [])] for name in split_names}

    manifest = {
        "dataset_name": out_cfg["dataset_name"],
        "data": {
            "raw": raw_cfg,
            "processing": proc_cfg,
            "split": split_cfg,
            "output": out_cfg,
            "stats_path": str(stats_path) if stats_path is not None else None,
        },
        "processor": "mesh_gnn",
        "split_config": split_cfg,
        "split_names": split_names,
        "split_shapes": split_shapes,
        "split_shards": split_shards,
        "stats_split": stats_split,
        "stats_path": str(stats_path) if stats_path is not None else None,
        "has_stats": True,
        "grid": {
            "lat": lat_vals.tolist(),
            "lon": lon_vals.tolist(),
            "level": np.asarray(ds[level_dim].values, dtype=np.float32).reshape(-1).tolist(),
        },
        "mesh_gnn": {
            "mesh_splits": mesh_splits,
            "grid_num_nodes": int(static_graph["grid"]["num_nodes"]),
            "mesh_num_nodes": int(static_graph["mesh"]["num_nodes"]),
            "grid2mesh_num_edges": int(static_graph["grid2mesh"]["num_edges"]),
            "mesh_num_edges": int(static_graph["mesh_graph"]["num_edges"]),
            "mesh2grid_num_edges": int(static_graph["mesh2grid"]["num_edges"]),
            "node_feature_names": list(static_graph["feature_schema"]["node_features"]),
            "edge_feature_names": list(static_graph["feature_schema"]["edge_features"]),
            "edge_feature_mode": str(static_graph["feature_schema"]["edge_feature_mode"]),
        },
        "dynamic_features": {
            "vars": variables,
            "level_dim": level_dim,
            "selected_levels": (feature_meta or {}).get("selected_levels", []),
            "dynamic_feature_order": (feature_meta or {}).get("dynamic_feature_order", "variable-major, then level-major"),
            "num_dynamic_features": int(feature_shape[-1]) if feature_shape is not None else 0,
        },
        "artifacts": {
            "split_shards": split_shards,
            "stats": "stats.pt",
            "static_graph": "mesh_gnn_static.pt",
        },
    }
    dump_yaml_mapping({"format_version": 1, **manifest}, processed_dir / "manifest.yaml")
    return manifest, split_shapes
