use cgmath::*;

use std::mem;
use std::ops::{Index, IndexMut};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct FaceIndex(pub u8);

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct VertexIndex(pub u8);

pub struct Mesh {
    pub verts: Vec<Vertex>,
    pub faces: Vec<Face>,
}

impl Mesh {
    pub fn get_vertex_face_ids(&self, vertex_id: VertexIndex, buf: &mut Vec<FaceIndex>) {
        for (index, face) in self.faces.iter().enumerate() {
            if face.has_vertex(vertex_id) {
                buf.push(FaceIndex(index as u8));
            }
        }
    }

    pub fn get_edge_face_ids(&self, edge: Edge) -> [FaceIndex; 2] {
        let mut face_ids = [FaceIndex::default(); 2];
        let mut i = 0;
        for (index, face) in self.faces.iter().enumerate() {
            if face.has_edge(edge) {
                face_ids[i] = FaceIndex(index as u8);
                i += 1;
                if i == 2 { return face_ids; }
            }
        }
        panic!("failed to find two faces adjacent to edge: {:?}", edge);
    }

    pub fn get_edge_verts(&self, edge: Edge) -> [Vertex; 2] {
        [self[edge.0], self[edge.1]]
    }

    pub fn get_edge_points(&self, edge: Edge) -> [Point3<f32>; 2] {
        [self[edge.0].pos, self[edge.1].pos]
    }

    pub fn get_face_verts(&self, face: Face) -> [Vertex; 3] {
        [self[face.0], self[face.1], self[face.2]]
    }

    pub fn get_face_points(&self, face: Face) -> [Point3<f32>; 3] {
        [self[face.0].pos, self[face.1].pos, self[face.2].pos]
    }
}

impl Index<FaceIndex> for Mesh {
    type Output = Face;

    #[inline]
    fn index(&self, index: FaceIndex) -> &Self::Output {
        &self.faces[index.0 as usize]
    }
}

impl IndexMut<FaceIndex> for Mesh {
    #[inline]
    fn index_mut(&mut self, index: FaceIndex) -> &mut Self::Output {
        &mut self.faces[index.0 as usize]
    }
}

impl Index<VertexIndex> for Mesh {
    type Output = Vertex;

    #[inline]
    fn index(&self, index: VertexIndex) -> &Self::Output {
        &self.verts[index.0 as usize]
    }
}

impl IndexMut<VertexIndex> for Mesh {
    #[inline]
    fn index_mut(&mut self, index: VertexIndex) -> &mut Self::Output {
        &mut self.verts[index.0 as usize]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Face (
    pub VertexIndex, pub VertexIndex, pub VertexIndex,
);

impl Face {
    fn edges(&self) -> [Edge; 3] {
        [
            Edge(self.0, self.1),
            Edge(self.1, self.2),
            Edge(self.2, self.0),
        ]
    }

    // CLEANUP: better API for this kind of action
    pub fn local_index_of(&self, target_id: VertexIndex) -> usize {
        for (index, &vertex_id) in self.as_ref().iter().enumerate() {
            if target_id == vertex_id { return index; }
        }
        panic!("target vertex id not found in face");
    }

    #[inline]
    pub fn has_edge(&self, edge: Edge) -> bool {
        self.has_vertex(edge.0) && self.has_vertex(edge.1)
    }

    #[inline]
    pub fn has_vertex(&self, vertex_id: VertexIndex) -> bool {
        self.0 == vertex_id || self.1 == vertex_id || self.2 == vertex_id
    }

    pub fn vertex_opposite(&self, edge: Edge) -> VertexIndex {
        assert!(self.has_edge(edge), "Face does not contain given edge");
        for &vertex_id in self.as_ref().iter() {
            if !edge.has_vertex(vertex_id) { return vertex_id; }
        }
        panic!("Face contains duplicate vertices");
    }

    pub fn edge_opposite(&self, vertex_id: VertexIndex) -> Edge {
        let slice = self.as_ref();
        for i in 0..3 {
            if slice[i] == vertex_id {
                return Edge(slice[(i+1)%3], slice[(i+2)%3]);
            }
        }
        panic!("Face does not contain given vertex");
    }

    pub fn edges_opposite(&self, edge: Edge) -> [Edge; 2] {
        let apex = self.vertex_opposite(edge);
        [Edge(edge.0, apex), Edge(apex, edge.1)] // TODO: check vertex order
    }
}

impl AsRef<[VertexIndex; 3]> for Face {
    fn as_ref(&self) -> &[VertexIndex; 3] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[VertexIndex; 3]> for Face {
    fn as_mut(&mut self) -> &mut [VertexIndex; 3] {
        unsafe { mem::transmute(self) }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Edge(pub VertexIndex, pub VertexIndex);

impl Edge {
    #[inline]
    pub fn has_vertex(&self, vertex_id: VertexIndex) -> bool {
        self.0 == vertex_id || self.1 == vertex_id
    }
}

impl AsRef<[VertexIndex; 2]> for Edge {
    #[inline]
    fn as_ref(&self) -> &[VertexIndex; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl AsMut<[VertexIndex; 2]> for Edge {
    #[inline]
    fn as_mut(&mut self) -> &mut [VertexIndex; 2] {
        unsafe { mem::transmute(self) }
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Edge) -> bool {
        (self.0 == other.0 && self.1 == other.1) ||
        (self.0 == other.1 && self.1 == other.0)
    }
}
impl Eq for Edge {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Point3<f32>,
    pub color: Vector3<f32>,
}

impl Vertex {
    #[inline]
    pub fn at(pos: impl Into<Point3<f32>>) -> Vertex {
        Vertex {
            pos: pos.into(),
            color: Vector3 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            pos: Point3 { x: 0.0, y: 0.0, z: 0.0 },
            color: Vector3 { x: 0.0, y: 0.0, z: 0.0 },
        }
    }
}
