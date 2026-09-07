use async_std::task::{self, block_on};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;

use line_drawing::{VoxelOrigin, WalkVoxels};
use ordered_float::NotNan;
use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::math::{Isometry, Point, Vector};
use parry3d_f64::query::{intersection_test, PointQuery};
use parry3d_f64::shape::{Cuboid, Shape, TriMesh, Triangle};

use crate::squarion::*;
use crate::svo::*;

#[derive(Debug, Clone, PartialEq)]
enum Voxel {
    Internal,
    External,
    Boundry(bool),
}

fn voxelize(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
    extent: usize,
    clip_range: &RangeZYX,
) -> Svo<Voxel> {
    let voxel_size = aabb.extents().x / extent as f64;
    Svo::from_fn(origin, extent, &|range| {
        if range.intersection(clip_range).volume() == 0 {
            return SvoReturn::Leaf(Voxel::External);
        }

        let mins = aabb.mins + (range.origin - origin).map(|v| v as f64) * voxel_size;
        let maxs = mins + range.size.map(|v| v as f64) * voxel_size;
        let aabb = Aabb::new(mins, maxs);

        // Scale up the region slightly. Makes intersection detection more robust.
        let cuboid = Cuboid::new(aabb.half_extents() * 1.05);
        let cuboid_pos = Isometry::from(aabb.center());
        if !intersection_test(isometry, mesh, &cuboid_pos, &cuboid).unwrap() {
            // Vote on if the voxel is inside or outside. We need to do this because some people won't
            // read the FAQ, and try to import non-manifold meshes. This makes the process more reliable.
            let mut inside_count = mesh.contains_point(isometry, &aabb.center()) as u32;
            for point in aabb.vertices() {
                inside_count += mesh.contains_point(isometry, &point) as u32
            }
            // Bias towards assuming outside, since it's better to have empty internals than random
            // floating cubes.
            if inside_count >= 7 {
                SvoReturn::Leaf(Voxel::Internal)
            } else {
                SvoReturn::Leaf(Voxel::External)
            }
        } else if range.volume() == 1 {
            // We do a quick check to see if the voxel is "significant", i.e. the center is in the mesh.
            //
            // This helps remove artifacts from internal angles in the model.
            let significant = mesh.contains_point(isometry, &aabb.center());
            SvoReturn::Leaf(Voxel::Boundry(significant))
        } else {
            SvoReturn::Internal(Voxel::Boundry(false))
        }
    })
}

fn discretize(point: Point<f64>, voxel_size: f64) -> Point<f64> {
    (84.0 * point / voxel_size).map(|v| v.round())
}

// In game voxels operate on a discrete grid, so the best solutions are ones that
// minimize error when going from the model surface to the in game voxel surface.
fn lowest_error_point_on_surface(
    starts: &[Point<f64>],
    end: &Point<f64>,
    shape: &impl Shape,
) -> Point<f64> {
    let mut lowest_error = f64::MAX;
    let mut best = *end;

    for start in starts {
        if (end - start).magnitude() < 0.1 {
            continue;
        }
        let dir = (end - start).normalize();
        let start = end - 5.0 * dir;
        let end = end + 5.0 * dir;
        for (x, y, z) in WalkVoxels::<f64, i64>::new(
            (start.x, start.y, start.z),
            (end.x, end.y, end.z),
            &VoxelOrigin::Corner,
        ) {
            let point = Point::new(x as f64, y as f64, z as f64);
            let error = shape.distance_to_local_point(&point, false);
            if error < lowest_error {
                lowest_error = error;
                best = point;
            }
        }
    }
    best
}

fn to_voxel_offset(offset: Vector<f64>) -> Vector<u8> {
    offset.map(|v| (126.0 + v).clamp(0.0, 252.0) as u8)
}

fn to_full_offset(offset: Vector<f64>) -> Vector<u8> {
    offset.map(|v| (126.0 + v.round()).clamp(0.0, 252.0) as u8)
}

// Closest point on the OBJ, clamped to this corner's cell.
fn closest_surface_offset(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    corner: Point<f64>,
    voxel_size: f64,
) -> Vector<u8> {
    let projected = mesh.project_point(isometry, &corner, false).point;
    to_full_offset((projected - corner) * 84.0 / voxel_size)
}

// Tries to snap to the nearest vertex, then edge, then face.
//
// Results in some artifacts in game where too many voxels snap to a vertex/edge and as a result
// you get some weird shadows on flat surfaces, but this otherwise preserves the model features.
fn calculate_vertex_offset(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    anchor: Point<f64>,
    voxel_size: f64,
) -> Vector<u8> {
    let discrete_anchor = discretize(anchor, voxel_size);
    let discrete_pos = discretize(aabb.center(), voxel_size);
    let (_, feature) = mesh.project_point_and_get_feature(isometry, &anchor);
    let face = feature.unwrap_face();
    let original_triangle = mesh.triangle(face).transformed(isometry);
    let a = discretize(original_triangle.a, voxel_size);
    let b = discretize(original_triangle.b, voxel_size);
    let c = discretize(original_triangle.c, voxel_size);
    let triangle = Triangle::new(a, b, c);

    let closest_vertex = triangle
        .vertices()
        .iter()
        .min_by_key(|v| NotNan::new((*v - discrete_anchor).magnitude()).unwrap())
        .unwrap();
    let offset = closest_vertex - discrete_pos;
    if offset.magnitude() < 84.0 {
        to_voxel_offset(closest_vertex - discrete_pos)
    } else {
        let (segment, closest_edge) = triangle
            .edges()
            .map(|s| (s, s.project_local_point(&discrete_anchor, false).point))
            .into_iter()
            .min_by_key(|(_, v)| NotNan::new((*v - discrete_anchor).magnitude()).unwrap())
            .unwrap();
        let offset = closest_edge - discrete_pos;
        if offset.magnitude() < 84.0 {
            let best = lowest_error_point_on_surface(&[segment.a], &closest_edge, &segment);
            to_voxel_offset(best - discrete_pos)
        } else {
            // Detect a degenerate triangle.
            if triangle.area() <= 1e-6 {
                let fallback = original_triangle
                    .project_local_point(&discrete_anchor, false)
                    .point;
                to_voxel_offset(fallback - discrete_pos)
            } else {
                let point = triangle.project_local_point(&discrete_anchor, false).point;
                let best = lowest_error_point_on_surface(&[a, b, c], &point, &triangle);
                to_voxel_offset(best - discrete_pos)
            }
        }
    }
}

fn grid_corner_world(
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
    point: Point<i32>,
) -> Point<f64> {
    aabb.mins + voxel_size * (point - origin).map(|v| v as f64)
}

fn mesh_normal(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    point: Point<f64>,
) -> Option<Vector<f64>> {
    let (_, feature) = mesh.project_point_and_get_feature(isometry, &point);
    mesh.triangle(feature.unwrap_face())
        .transformed(isometry)
        .normal()
        .map(|n| *n)
}

fn collect_solid_cells(voxels: &Svo<Voxel>) -> HashSet<Point<i32>> {
    let mut solid = HashSet::new();
    voxels.cata(|range, v, cs| {
        if cs.is_some() {
            return;
        }
        if matches!(v, Voxel::Internal | Voxel::Boundry(true)) {
            solid.insert(range.origin);
        }
    });
    solid
}

const FILL_NEIGHBORS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

// Face fills sit on one OBJ plane and can be flattened onto it. Edge/corner
// fills see two surfaces; those leftover corners are snapped to neighbors.
fn is_face_fill(
    cell: Point<i32>,
    solid: &HashSet<Point<i32>>,
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
) -> bool {
    let mut toward_solid: Option<Vector<f64>> = None;
    let mut solid_neighbors = 0u32;
    for n in &FILL_NEIGHBORS {
        let neighbor = cell + Vector::new(n[0], n[1], n[2]);
        if solid.contains(&neighbor) {
            solid_neighbors += 1;
            toward_solid = Some(Vector::new(n[0] as f64, n[1] as f64, n[2] as f64));
        }
    }
    if solid_neighbors != 1 {
        return false;
    }

    let center = aabb.mins + voxel_size * (cell - origin).map(|v| v as f64 + 0.5);
    let Some(n0) = mesh_normal(isometry, mesh, center) else {
        return false;
    };
    if n0.dot(&toward_solid.unwrap()) < 0.3 {
        return false;
    }
    for offset in &RangeZYX::OFFSETS {
        let corner = grid_corner_world(
            aabb,
            origin,
            voxel_size,
            cell + Vector::from_row_slice(offset),
        );
        let Some(n) = mesh_normal(isometry, mesh, corner) else {
            continue;
        };
        if n0.dot(&n) < 0.5 {
            return false;
        }
    }
    true
}

fn encoded_world(
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
    point: Point<i32>,
    encoded: Point<u8>,
) -> Point<f64> {
    let corner = grid_corner_world(aabb, origin, voxel_size, point);
    corner + encoded.coords.map(|c| (c as f64 - 126.0) / 84.0 * voxel_size)
}

fn encode_from_rest(
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
    point: Point<i32>,
    world: Point<f64>,
) -> Point<u8> {
    let rest = grid_corner_world(aabb, origin, voxel_size, point);
    Point::origin() + to_full_offset((world - rest) * 84.0 / voxel_size)
}

fn neighbor_average(
    result: &HashMap<Point<i32>, Point<u8>>,
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
    point: Point<i32>,
) -> Option<Point<f64>> {
    let mut sum = Vector::zeros();
    let mut count = 0u32;
    for n in &FILL_NEIGHBORS {
        let neighbor = point + Vector::new(n[0], n[1], n[2]);
        if let Some(enc) = result.get(&neighbor) {
            sum += encoded_world(aabb, origin, voxel_size, neighbor, *enc).coords;
            count += 1;
        }
    }
    if count >= 2 {
        Some(Point::origin() + sum / count as f64)
    } else {
        None
    }
}

fn snap_unset_to_neighbors(
    voxels: &Svo<Voxel>,
    result: &mut HashMap<Point<i32>, Point<u8>>,
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
) {
    let mut unset = HashSet::new();
    voxels.cata(|range, v, cs| {
        if cs.is_some() {
            return;
        }
        let Voxel::Boundry(false) = v else {
            return;
        };
        for offset in &RangeZYX::OFFSETS {
            let point = range.origin + Vector::from_row_slice(offset);
            if !result.contains_key(&point) {
                unset.insert(point);
            }
        }
    });

    loop {
        let mut placed = Vec::new();
        for point in &unset {
            if let Some(target) = neighbor_average(result, aabb, origin, voxel_size, *point) {
                result.insert(
                    *point,
                    encode_from_rest(aabb, origin, voxel_size, *point, target),
                );
                placed.push(*point);
            }
        }
        if placed.is_empty() {
            break;
        }
        for point in placed {
            unset.remove(&point);
        }
    }

    for point in unset {
        let rest = grid_corner_world(aabb, origin, voxel_size, point);
        let target = mesh.project_point(isometry, &rest, false).point;
        result.insert(
            point,
            encode_from_rest(aabb, origin, voxel_size, point, target),
        );
    }
}

fn relax_surface_vertices(
    vertices: &mut HashMap<Point<i32>, Point<u8>>,
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
    iterations: u32,
) {
    for _ in 0..iterations {
        let previous = vertices.clone();
        for (point, encoded) in previous.iter() {
            let self_pos = encoded_world(aabb, origin, voxel_size, *point, *encoded);
            let mut sum = Vector::zeros();
            let mut count = 0u32;
            for n in &FILL_NEIGHBORS {
                let neighbor = *point + Vector::new(n[0], n[1], n[2]);
                if let Some(enc) = previous.get(&neighbor) {
                    sum += encoded_world(aabb, origin, voxel_size, neighbor, *enc).coords;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let blended = self_pos + ((sum / count as f64) - self_pos.coords) * 0.5;
            let projected = mesh.project_point(isometry, &blended, false).point;
            vertices.insert(
                *point,
                encode_from_rest(aabb, origin, voxel_size, *point, projected),
            );
        }
    }
}

fn flatten_fill_cell(
    range: &RangeZYX,
    result: &mut HashMap<Point<i32>, Point<u8>>,
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
    voxel_size: f64,
) {
    let center = aabb.mins + voxel_size * (range.origin - origin).map(|v| v as f64 + 0.5);
    let (proj, feature) = mesh.project_point_and_get_feature(isometry, &center);
    let tri = mesh.triangle(feature.unwrap_face()).transformed(isometry);
    let Some(normal) = tri.normal() else {
        return;
    };
    let nrm = *normal;
    let plane_p = proj.point;
    for offset in &RangeZYX::OFFSETS {
        let point = range.origin + Vector::from_row_slice(offset);
        if result.contains_key(&point) {
            continue;
        }
        let rest = grid_corner_world(aabb, origin, voxel_size, point);
        let projected = rest - nrm * nrm.dot(&(rest.coords - plane_p.coords));
        result.insert(
            point,
            Point::origin() + to_full_offset((projected - rest) * 84.0 / voxel_size),
        );
    }
}

fn extract_vertices(
    voxels: &Svo<Voxel>,
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    origin: Point<i32>,
    smooth: u32,
) -> HashMap<Point<i32>, Point<u8>> {
    let voxel_size = aabb.extents().x / voxels.range.size.x as f64;
    let mut significant_points = HashMap::new();
    voxels.cata(|range, v, cs| {
        if cs.is_some() {
            return;
        }
        match v {
            Voxel::Boundry(significant) => {
                assert_eq!(range.volume(), 1);
                let center =
                    aabb.mins + voxel_size * (range.origin - origin).map(|v| v as f64 + 0.5);
                for offset in &RangeZYX::OFFSETS {
                    let offset = Vector::from_row_slice(offset);
                    let point = range.origin + offset;

                    let pos = aabb.mins + voxel_size * (point - origin).map(|v| v as f64);
                    if mesh.contains_point(isometry, &pos) != *significant {
                        let entry = significant_points
                            .entry(point)
                            .or_insert_with(|| Vec::new());
                        entry.push(center.coords)
                    }
                }
            }
            _ => (),
        };
    });

    let mut result = HashMap::new();
    for (point, anchors) in significant_points {
        let pos = grid_corner_world(aabb, origin, voxel_size, point);
        let best = if smooth >= 1 {
            closest_surface_offset(isometry, mesh, pos, voxel_size)
        } else {
            let anchor =
                anchors.iter().fold(Point::origin(), |a, v| a + v) / anchors.len() as f64;
            let aabb = Aabb::from_half_extents(pos, Vector::repeat(voxel_size * 1.5));
            calculate_vertex_offset(isometry, mesh, &aabb, anchor, voxel_size)
        };
        result.insert(point, Point::origin() + best);
    }

    if smooth >= 1 {
        let solid = collect_solid_cells(voxels);
        voxels.cata(|range, v, cs| {
            if cs.is_some() {
                return;
            }
            let Voxel::Boundry(false) = v else {
                return;
            };
            if !is_face_fill(
                range.origin,
                &solid,
                isometry,
                mesh,
                aabb,
                origin,
                voxel_size,
            ) {
                return;
            }
            flatten_fill_cell(&range, &mut result, isometry, mesh, aabb, origin, voxel_size);
        });
        snap_unset_to_neighbors(
            voxels,
            &mut result,
            isometry,
            mesh,
            aabb,
            origin,
            voxel_size,
        );
        if smooth >= 2 {
            relax_surface_vertices(
                &mut result,
                isometry,
                mesh,
                aabb,
                origin,
                voxel_size,
                smooth - 1,
            );
        }
    }
    result
}

// This is by far the most expensive part, mostly due to Trimesh being kinda slow and the algorithm itself
// being pretty naive. For now we just throw threads at it, but it can definitely be improved.
fn voxelize_chunk(
    isometry: &Isometry<f64>,
    mesh: &TriMesh,
    aabb: &Aabb,
    voxel_origin: &Point<i32>,
    material: u64,
    is_lod: bool,
    smooth: u32,
) -> Option<VoxelCellData> {
    // We have to over-voxelize that chunk due to the boundries expected in voxel cell data.
    // e.g. for an inner_range of [0, 0, 0] -> [32, 32, 32] the actual range of the chunk is
    //  [-1, -1, -1] -> [34, 34, 34], likely to remove seams when generating the mesh.
    let voxel_size = aabb.extents().x / 32.0;
    let voxel_size_offset = Vector::repeat(voxel_size);
    let origin = aabb.mins - voxel_size_offset * 2.0;

    let range = RangeZYX::with_extent(voxel_origin - Vector::repeat(1), 35);

    // Note that this large aabb could result in a lot of wasted computation, so we clip the range.
    let svo_aabb = Aabb::new(origin, origin + voxel_size_offset * 64.0);
    let voxels = voxelize(
        isometry,
        mesh,
        &svo_aabb,
        voxel_origin - Vector::repeat(2),
        64,
        &range,
    );

    let inner_range = RangeZYX::with_extent(*voxel_origin, 32);
    let mut grid = VertexGrid::new(range, inner_range);
    voxels.cata(|subrange, value, cs| {
        if cs.is_some() {
            return;
        }
        let (place_materials, place_positions) = match value {
            Voxel::External => (false, false),
            Voxel::Internal => (true, true),
            Voxel::Boundry(significant) => (smooth >= 1 || *significant, true),
        };
        if place_materials {
            // Materials are placed on the +[1, 1, 1] vertex.
            let material_range = RangeZYX {
                origin: subrange.origin + Vector::repeat(1),
                size: subrange.size,
            };
            grid.set_materials(&material_range, VertexMaterial::new(2));
        }
        if place_positions {
            // Set the default positions for all voxels. We will update the significant ones later.
            let voxel_range = RangeZYX {
                origin: subrange.origin,
                size: subrange.size + Vector::repeat(1),
            };
            grid.set_voxels(&voxel_range, VertexVoxel::new([126, 126, 126]));
        }
    });

    if !is_lod && grid.is_empty() {
        return None;
    }

    // Extract the non-default vertices and set them now.
    let vertices = extract_vertices(
        &voxels,
        isometry,
        mesh,
        &svo_aabb,
        voxel_origin - Vector::repeat(2),
        smooth,
    );
    for (point, offset) in vertices {
        grid.set_voxel(&point, VertexVoxel::new([offset.x, offset.y, offset.z]));
    }

    let mut mapping = MaterialMapper::default();

    // Every blueprint I checked had this debug material in the first index.
    // I assume there is a reason for it, so we'll add it as well.
    mapping.insert(
        1,
        MaterialId {
            id: 157903047,
            short_name: "Debug1\0\0".into(),
        },
    );
    mapping.insert(
        2,
        MaterialId {
            id: material,
            short_name: "Material".into(),
        },
    );

    Some(VoxelCellData::new(grid, mapping))
}

pub struct Voxelizer {
    isometry: Arc<Isometry<f64>>,
    mesh: Arc<TriMesh>,
}

impl Voxelizer {
    pub fn new(isometry: Isometry<f64>, mesh: TriMesh) -> Voxelizer {
        Voxelizer {
            isometry: Arc::new(isometry),
            mesh: Arc::new(mesh),
        }
    }

    pub fn create_lods(
        &self,
        aabb: &Aabb,
        origin: Point<i32>,
        height: usize,
        material: u64,
        smooth: u32,
    ) -> Svo<Option<VoxelCellData>> {
        let extent = 1 << height;
        let chunk_size = aabb.extents().x / extent as f64;
        let chunk_futures = Svo::from_fn(origin, extent, &|range| {
            let mins = aabb.mins + (range.origin - origin).map(|v| v as f64) * chunk_size;
            let maxs = mins + range.size.map(|v| v as f64) * chunk_size;
            let aabb = Aabb::new(mins.into(), maxs.into());

            let cuboid = Cuboid::new(aabb.half_extents() * 1.05);
            let cuboid_pos = Isometry::from(aabb.center());
            if intersection_test(&self.isometry, self.mesh.as_ref(), &cuboid_pos, &cuboid).unwrap()
            {
                let is_lod = range.size.x > 1;
                let voxel_origin = range.origin * 32 / range.size.x;
                let isometry = self.isometry.clone();
                let mesh = self.mesh.clone();
                let task = task::spawn(async move {
                    voxelize_chunk(
                        &isometry,
                        &mesh,
                        &aabb,
                        &voxel_origin,
                        material,
                        is_lod,
                        smooth,
                    )
                });
                if range.size.x == 1 {
                    SvoReturn::Leaf(Some(task))
                } else {
                    SvoReturn::Internal(Some(task))
                }
            } else {
                SvoReturn::Leaf(None)
            }
        });

        chunk_futures.into_map(|f| f.map(|f| block_on(f)).flatten())
    }
}
