//! Static BVH over a set of spheres, built once (not per frame).
//!
//! Mirrors the JS `buildBVH` in `renderer.js` (used for imported-model
//! triangles) — same iterative median-split build, same depth-first node
//! layout with escape links so the WGSL traversal is a single stackless loop.
//! Sharing that layout means the ray tracer can walk this tree with the exact
//! same `BvhNode` struct and loop it already uses for model triangles.
//!
//! The cube field's instances never move after `init_scene` places them (only
//! their rotation animates), so this tree is built once and re-uploaded to
//! the GPU only when the scene is (re)built — never per frame.

use crate::math::Vec3;

/// Spheres per leaf. Small leaves keep the per-ray sphere test count low;
/// this matches the model BVH's `RT_LEAF_TRIS` in renderer.js.
const LEAF_COUNT: usize = 4;
/// Floats per node: bmin(vec4: xyz + escape) + bmax(vec4: xyz + leaf_count)
/// + first(vec4: x = first sphere index, yzw unused). Matches WGSL `BvhNode`.
pub const NODE_FLOATS: usize = 12;
/// Floats per reordered sphere record: center.xyz + radius.
pub const SPHERE_FLOATS: usize = 4;

struct Node {
    mn: [f32; 3],
    mx: [f32; 3],
    first: usize,
    count: usize, // 0 = internal, >0 = leaf sphere count
    escape: i32,
}

pub struct Build {
    pub nodes: Vec<f32>,
    pub spheres: Vec<f32>,
    pub node_count: usize,
}

/// Build a stackless BVH over `centers`/`radii` (same length). Returns the
/// flattened node array, the sphere records reordered to match the BVH's
/// leaf layout (so a leaf's `[first, first+count)` range is contiguous), and
/// the node count.
pub fn build(centers: &[Vec3], radii: &[f32]) -> Build {
    let n = centers.len();
    if n == 0 {
        return Build { nodes: Vec::new(), spheres: Vec::new(), node_count: 0 };
    }

    let bounds: Vec<([f32; 3], [f32; 3])> = centers
        .iter()
        .zip(radii.iter())
        .map(|(c, &r)| ([c.x - r, c.y - r, c.z - r], [c.x + r, c.y + r, c.z + r]))
        .collect();

    let mut order: Vec<usize> = (0..n).collect();
    let mut nodes: Vec<Node> = Vec::new();
    // Right child pushed first so the left child (pushed last) pops next and
    // becomes `self + 1` — the depth-first layout the escape links rely on.
    let mut stack: Vec<(usize, usize)> = vec![(0, n)];

    while let Some((start, end)) = stack.pop() {
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for &idx in &order[start..end] {
            let (bmn, bmx) = bounds[idx];
            for k in 0..3 {
                if bmn[k] < mn[k] { mn[k] = bmn[k]; }
                if bmx[k] > mx[k] { mx[k] = bmx[k]; }
            }
        }

        let count = end - start;
        if count <= LEAF_COUNT {
            nodes.push(Node { mn, mx, first: start, count, escape: -1 });
            continue;
        }

        // Split on the longest centroid-extent axis at its midpoint; fall
        // back to a median split if every centroid coincides.
        let mut cmn = [f32::INFINITY; 3];
        let mut cmx = [f32::NEG_INFINITY; 3];
        for &idx in &order[start..end] {
            let c = centers[idx];
            let p = [c.x, c.y, c.z];
            for k in 0..3 {
                if p[k] < cmn[k] { cmn[k] = p[k]; }
                if p[k] > cmx[k] { cmx[k] = p[k]; }
            }
        }
        let mut axis = 0usize;
        let mut ext = cmx[0] - cmn[0];
        if cmx[1] - cmn[1] > ext { axis = 1; ext = cmx[1] - cmn[1]; }
        if cmx[2] - cmn[2] > ext { axis = 2; ext = cmx[2] - cmn[2]; }

        let mid = if ext <= 1e-9 {
            (start + end) / 2
        } else {
            let split_pos = 0.5 * (cmn[axis] + cmx[axis]);
            let axis_val = |idx: usize| -> f32 {
                let c = centers[idx];
                [c.x, c.y, c.z][axis]
            };
            let mut i = start as isize;
            let mut j = end as isize - 1;
            while i <= j {
                while i <= j && axis_val(order[i as usize]) < split_pos { i += 1; }
                while i <= j && axis_val(order[j as usize]) >= split_pos { j -= 1; }
                if i < j {
                    order.swap(i as usize, j as usize);
                }
            }
            let m = i as usize;
            if m == start || m == end { (start + end) / 2 } else { m }
        };

        nodes.push(Node { mn, mx, first: 0, count: 0, escape: -1 }); // internal
        stack.push((mid, end));
        stack.push((start, mid));
    }

    // Escape links: this preorder layout makes each node's subtree
    // contiguous, so escape[i] = i + subtree_size[i] (or -1 past the end).
    let node_count = nodes.len();
    let mut size = vec![0i32; node_count];
    for i in (0..node_count).rev() {
        if nodes[i].count > 0 {
            size[i] = 1;
        } else {
            let left = i + 1;
            let left_end = left + size[left] as usize;
            size[i] = 1 + size[left] + size[left_end];
        }
    }
    for i in 0..node_count {
        let esc = i as i32 + size[i];
        nodes[i].escape = if esc as usize >= node_count { -1 } else { esc };
    }

    let mut flat = vec![0.0f32; node_count * NODE_FLOATS];
    for (i, nd) in nodes.iter().enumerate() {
        let o = i * NODE_FLOATS;
        flat[o] = nd.mn[0];
        flat[o + 1] = nd.mn[1];
        flat[o + 2] = nd.mn[2];
        flat[o + 3] = nd.escape as f32;
        flat[o + 4] = nd.mx[0];
        flat[o + 5] = nd.mx[1];
        flat[o + 6] = nd.mx[2];
        flat[o + 7] = nd.count as f32;
        flat[o + 8] = nd.first as f32;
    }

    let mut spheres = vec![0.0f32; n * SPHERE_FLOATS];
    for (i, &idx) in order.iter().enumerate() {
        let o = i * SPHERE_FLOATS;
        let c = centers[idx];
        spheres[o] = c.x;
        spheres[o + 1] = c.y;
        spheres[o + 2] = c.z;
        spheres[o + 3] = radii[idx];
    }

    Build { nodes: flat, spheres, node_count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3;
    use std::collections::BTreeSet;

    fn to_bits(v: [f32; 4]) -> [u32; 4] {
        [v[0].to_bits(), v[1].to_bits(), v[2].to_bits(), v[3].to_bits()]
    }

    fn spheres_set(spheres: &[f32]) -> BTreeSet<[u32; 4]> {
        spheres
            .chunks_exact(SPHERE_FLOATS)
            .map(|c| to_bits([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn empty_scene_has_no_nodes() {
        let b = build(&[], &[]);
        assert_eq!(b.node_count, 0);
        assert!(b.nodes.is_empty());
        assert!(b.spheres.is_empty());
    }

    #[test]
    fn single_sphere_is_one_leaf() {
        let centers = [vec3(1.0, 2.0, 3.0)];
        let radii = [0.5f32];
        let b = build(&centers, &radii);
        assert_eq!(b.node_count, 1);
        assert_eq!(b.nodes[7], 1.0); // leaf count
        assert_eq!(b.nodes[3], -1.0); // terminal escape
    }

    // The reordered sphere buffer must be a permutation of the input: every
    // sphere appears exactly once, none duplicated or dropped.
    #[test]
    fn reordered_spheres_are_a_permutation() {
        let mut seed = 12345u32;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32) / (u32::MAX as f32)
        };
        let centers: Vec<Vec3> = (0..137)
            .map(|_| vec3(rng() * 100.0, rng() * 100.0, rng() * 100.0))
            .collect();
        let radii: Vec<f32> = (0..137).map(|_| 0.2 + rng()).collect();

        let mut expected = vec![0.0f32; 137 * SPHERE_FLOATS];
        for (i, (c, &r)) in centers.iter().zip(radii.iter()).enumerate() {
            let o = i * SPHERE_FLOATS;
            expected[o] = c.x;
            expected[o + 1] = c.y;
            expected[o + 2] = c.z;
            expected[o + 3] = r;
        }

        let b = build(&centers, &radii);
        assert_eq!(b.spheres.len(), expected.len());
        assert_eq!(spheres_set(&b.spheres), spheres_set(&expected));

        // Leaf ranges must exactly partition [0, n) with no gaps or overlaps.
        let mut covered = vec![false; 137];
        for node in b.nodes.chunks_exact(NODE_FLOATS) {
            let leaf_count = node[7] as usize;
            if leaf_count > 0 {
                let first = node[8] as usize;
                for k in first..first + leaf_count {
                    assert!(!covered[k], "sphere slot {k} covered by two leaves");
                    covered[k] = true;
                }
            }
        }
        assert!(covered.iter().all(|&c| c), "every slot must be covered by some leaf");
    }

    // The root node's AABB must contain every sphere's bounds.
    #[test]
    fn root_bounds_contain_all_spheres() {
        let centers = [
            vec3(-40.0, 5.0, 10.0),
            vec3(20.0, -8.0, -30.0),
            vec3(0.0, 0.0, 0.0),
            vec3(15.0, 15.0, 15.0),
            vec3(-5.0, -5.0, 40.0),
        ];
        let radii = [1.0, 2.5, 0.5, 3.0, 1.2];
        let b = build(&centers, &radii);
        let root = &b.nodes[0..NODE_FLOATS];
        let (rmn, rmx) = ([root[0], root[1], root[2]], [root[4], root[5], root[6]]);
        for (c, &r) in centers.iter().zip(radii.iter()) {
            assert!(c.x - r >= rmn[0] - 1e-4 && c.x + r <= rmx[0] + 1e-4);
            assert!(c.y - r >= rmn[1] - 1e-4 && c.y + r <= rmx[1] + 1e-4);
            assert!(c.z - r >= rmn[2] - 1e-4 && c.z + r <= rmx[2] + 1e-4);
        }
        // Root's subtree is the whole tree, so a miss on it must terminate.
        assert_eq!(root[3], -1.0);
    }
}
