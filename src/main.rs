extern crate gl;
use gl::types::*;

extern crate glutin;
use glutin::*;

extern crate cgmath;
use cgmath::*;

use std::default::Default;
use std::mem::{self, size_of};
use std::ops::{Index, IndexMut};
use std::os::raw::c_char;
use std::ptr;

macro_rules! cstr {
    ($s:expr) => (
        concat!($s, "\0") as *const str as *const [c_char] as *const c_char
    );
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct FaceIndex(u8);

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct VertexIndex(u8);

struct Mesh {
    verts: Box<[Vertex]>,
    faces: Box<[Face]>,
}

impl Mesh {
    fn get_vertex_face_ids(&self, vertex_id: VertexIndex, buf: &mut Vec<FaceIndex>) {
        for (index, face) in self.faces.iter().enumerate() {
            if face.has_vertex(vertex_id) {
                buf.push(FaceIndex(index as u8));
            }
        }
    }

    fn get_edge_face_ids(&self, edge: Edge) -> [FaceIndex; 2] {
        let mut face_ids = [FaceIndex::default(); 2];
        let mut i = 0;
        for (index, face) in self.faces.iter().enumerate() {
            if face.has_edge(edge) {
                face_ids[i] = FaceIndex(index as u8);
                i += 1;
                if i == 2 { return face_ids; }
            }
        }
        panic!("failed to find two faces adjacent to edge");
    }

    fn get_face_verts(&self, face: Face) -> [Vertex; 3] {
        [self[face.0], self[face.1], self[face.2]]
    }

    fn get_face_points(&self, face: Face) -> [Point3<f32>; 3] {
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
struct Face (
    VertexIndex, VertexIndex, VertexIndex,
);

impl Face {
    fn edges(&self) -> [Edge; 3] {
        [ Edge(self.0, self.1)
        , Edge(self.1, self.2)
        , Edge(self.2, self.0)
        ]
    }

    #[inline]
    fn has_edge(&self, edge: Edge) -> bool {
        self.has_vertex(edge.0) && self.has_vertex(edge.1)
    }

    #[inline]
    fn has_vertex(&self, vertex_id: VertexIndex) -> bool {
        self.0 == vertex_id || self.1 == vertex_id || self.2 == vertex_id
    }

    fn vertex_opposite(&self, edge: Edge) -> VertexIndex {
        assert!(self.has_edge(edge), "Face does not contain given edge");
        for &vertex_id in self.as_ref().iter() {
            if !edge.has_vertex(vertex_id) { return vertex_id; }
        }
        panic!("Face contains duplicate vertices");
    }

    fn edge_opposite(&self, vertex_id: VertexIndex) -> Edge {
        let slice = self.as_ref();
        for i in 0..3 {
            if slice[i] == vertex_id {
                return Edge(slice[(i+1)%3], slice[(i+2)%3]);
            }
        }
        panic!("Face does not contain given vertex");
    }

    fn edges_opposite(&self, edge: Edge) -> [Edge; 2] {
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
struct Edge(VertexIndex, VertexIndex);

impl Edge {
    #[inline]
    fn has_vertex(&self, vertex_id: VertexIndex) -> bool {
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
struct Vertex {
    pos: Point3<f32>,
    color: Vector3<f32>,
}

impl Vertex {
    #[inline]
    fn at(pos: impl Into<Point3<f32>>) -> Vertex {
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

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn main() {
    let mut events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title("Skyro")
        .with_dimensions(dpi::LogicalSize::new(WIDTH as _, HEIGHT as _));
    let context = ContextBuilder::new();
    let gl_window = GlWindow::new(window, context, &events_loop).unwrap();

    unsafe { gl_window.make_current().unwrap(); }
    gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

    #[repr(C)]
    struct Mvp {
        modelview: Matrix4<f32>,
        projection: Matrix4<f32>,
    };

    let (mut mvp, mvp_ubo, mvp_binding_index) = unsafe {
        let mut mvp = Mvp {
            modelview: Matrix4::identity(),
            projection: Matrix4::identity(),
        };

        let mvp_ubo = gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::UNIFORM_BUFFER, mvp_ubo);
        gl::BufferData(
            gl::UNIFORM_BUFFER,
            size_of::<Mvp>() as GLsizeiptr,
            ptr::null(),
            gl::DYNAMIC_DRAW,
        );

        let mvp_binding_index: GLuint = 1;
        gl::BindBufferBase(gl::UNIFORM_BUFFER, mvp_binding_index, mvp_ubo);

        (mvp, mvp_ubo, mvp_binding_index)
    };

    let (program, a_pos, a_color) = unsafe {
        let program = link_shaders(&[
            compile_shader(include_str!("shader/sky.vert.glsl"), gl::VERTEX_SHADER),
            compile_shader(include_str!("shader/sky.frag.glsl"), gl::FRAGMENT_SHADER),
        ]);
        gl::UseProgram(program);

        let a_pos   = gl::GetAttribLocation(program, cstr!("position")) as GLuint;
        let a_color = gl::GetAttribLocation(program, cstr!("vertex_color")) as GLuint;

        let mvp_index = gl::GetUniformBlockIndex(program, cstr!("Mvp"));
        gl::UniformBlockBinding(program, mvp_index, mvp_binding_index);

        (program, a_pos, a_color)
    };

    let (mut mesh, mesh_vao, mesh_vbo, mesh_ebo) = unsafe {
        let (mut verts, faces) = make_icosahedron();
        for mut vert in verts.iter_mut() {
            vert.color.x = vert.pos.x / 2.0 + 0.5;
            vert.color.y = vert.pos.y / 2.0 + 0.5;
            vert.color.z = vert.pos.z / 2.0 + 0.5;
        }

        let mesh = Mesh { verts: Box::new(verts), faces: Box::new(faces) };

        let vao = gen_object(gl::GenVertexArrays);
        gl::BindVertexArray(vao);

        let vbo = gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (size_of::<Vertex>() * verts.len()) as GLsizeiptr,
            mesh.verts.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW,
        );

        gl::EnableVertexAttribArray(a_pos);
        gl::VertexAttribPointer(
            a_pos, 3, gl::FLOAT, gl::FALSE,
            size_of::<Vertex>() as GLsizei, ptr::null() as *const GLvoid,
        );

        gl::EnableVertexAttribArray(a_color);
        gl::VertexAttribPointer(
            a_color, 3, gl::FLOAT, gl::FALSE,
            size_of::<Vertex>() as GLsizei, ptr::null::<f32>().offset(3) as *const GLvoid,
        );

        let ebo = gen_object(gl::GenBuffers);
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (size_of::<Face>() * faces.len()) as GLsizeiptr,
            mesh.faces.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW,
        );

        (mesh, vao, vbo, ebo)
    };

    let mut camera_fovy: Deg<f32> = Deg(60.0);
    let mut camera_distance: f32 = 3.0;
    let mut camera_elevation: Deg<f32> = Deg(0.0);
    let mut camera_azimuth: Deg<f32> = Deg(0.0);

    let mut mouse_pos = <Point2<f32>>::new(0.0, 0.0);
    let mut mouse_move: Option<Vector2<f32>> = None;
    let mut mouse_down = false;

    enum Selection {
        Face(FaceIndex),
        Vertex(VertexIndex, Vec<FaceIndex>),
        Edge(Edge, [FaceIndex; 2]),
    }
    impl PartialEq for Selection {
        fn eq(&self, other: &Selection) -> bool {
            match (self, other) {
                (Selection::Face(a),      Selection::Face(b))      => a == b,
                (Selection::Vertex(a, _), Selection::Vertex(b, _)) => a == b,
                (Selection::Edge(a, _),   Selection::Edge(b, _))   => a == b,
                _                                                  => false,
            }
        }
    }
    impl Eq for Selection {}

    let mut selection: Option<Selection> = None;
    let mut selection_drag = false;
    let mut selection_primary:   Vec<VertexIndex> = Vec::new();
    let mut selection_secondary: Vec<VertexIndex> = Vec::new();
    let mut selection_draw_mode: GLenum = gl::TRIANGLES;

    let mut need_render = true;
    let mut need_camera_update = true;
    let mut need_selection_update = true;

    let mut exit = false;
    while !exit {
        // collect input
        events_loop.poll_events(|event| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => { exit = true; },
                WindowEvent::MouseInput {
                    button: MouseButton::Left, state, ..
                } => match state {
                    ElementState::Pressed  => {
                        mouse_down = true;
                        if let Some(Selection::Vertex(..)) = selection {
                            selection_drag = true;
                        }
                    },
                    ElementState::Released => {
                        mouse_down = false;
                        selection_drag = false;
                    },
                },
                WindowEvent::CursorMoved { position: pos, .. } => {
                    let mouse_new_pos = Point2 {
                        x: pos.x as f32,
                        y: pos.y as f32,
                    };
                    mouse_move = mouse_move.or(Some(Vector2::new(0.0, 0.0)))
                        .map(|movement| movement + (mouse_new_pos - mouse_pos));
                    mouse_pos = mouse_new_pos;
                },
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_delta_x, delta_y), ..
                } => {
                    camera_distance -= delta_y * 0.2;
                    if camera_distance < 0.0 { camera_distance = 0.0; }
                    need_camera_update = true;
                },
                _ => (),
            },
            _ => (),
        });

        let mvp_inverse = (mvp.projection * mvp.modelview).inverse_transform()
            .expect("mvp inversion failed");

        let mouse_ray = {
            let mouse_ndc = Point2::<f32> {
                x: mouse_pos.x / WIDTH  as f32 *  2.0 - 1.0,
                y: mouse_pos.y / HEIGHT as f32 * -2.0 + 1.0,
            };

            let origin = mvp_inverse.transform_point(
                Point3::new(mouse_ndc.x, mouse_ndc.y, 0.0));
            let direction = (mvp_inverse.transform_point(
                Point3::new(mouse_ndc.x, mouse_ndc.y, 1.0)) - origin).normalize();

            Ray { origin, direction }
        };

        // handle mouse movement
        if let Some(movement) = mouse_move {
            if selection_drag {
                let vertex_id = if let Some(Selection::Vertex(vertex_id, ..)) = selection {
                    vertex_id
                } else { panic!("drag non-vertex selection"); };
                let ref mut vertex = mesh[vertex_id];

                let camera_aligned_plane = {
                    let center_back  = mvp_inverse.transform_point(Point3::new(0.0, 0.0, 1.0));
                    let center_front = mvp_inverse.transform_point(Point3::new(0.0, 0.0, 0.0));

                    let normal = (center_front - center_back).normalize();
                    let offset = normal.dot(vertex.pos.to_vec());

                    Plane { normal, offset }
                };

                vertex.pos = raycast_plane(camera_aligned_plane, mouse_ray)
                    .expect("bad camera plane");
                vertex.pos = Point3::from_vec(vertex.pos.to_vec().normalize());
                unsafe {
                    gl::BindBuffer(gl::ARRAY_BUFFER, mesh_vbo);
                    gl::BufferSubData(
                        gl::ARRAY_BUFFER,
                        (vertex_id.0 as usize * size_of::<Vertex>()) as GLintptr,
                        size_of::<Point3<f32>>() as GLsizeiptr,
                        vertex.pos.as_ptr() as *const GLvoid,
                    );
                }
                need_render = true;
            } else {
                if mouse_down {
                    camera_azimuth -= Deg(0.5 * movement.x);

                    camera_elevation += Deg(0.5 * movement.y);
                    if      camera_elevation < Deg(-85.0) { camera_elevation = Deg(-85.0); }
                    else if camera_elevation > Deg(85.0)  { camera_elevation = Deg(85.0);  }
                    need_camera_update = true;
                }
                need_selection_update = true;
            }
            mouse_move = None;
        }

        // update camera
        if need_camera_update {
            let camera_dir =
                ( Quaternion::from_angle_z(camera_azimuth)
                * Quaternion::from_angle_x(-camera_elevation)
                ).rotate_vector(Vector3::new(0.0, 1.0, 0.0));

            mvp.projection = perspective(camera_fovy, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);

            mvp.modelview = Matrix4::look_at(
                Point3::new(0.0, 0.0, 0.0) - camera_dir * camera_distance,
                Point3::new(0.0, 0.0, 0.0) + camera_dir,
                Vector3::new(0.0, 0.0, 1.0),
            );

            unsafe {
                gl::BindBuffer(gl::UNIFORM_BUFFER, mvp_ubo);
                gl::BufferData(
                    gl::UNIFORM_BUFFER,
                    size_of::<Mvp>() as GLsizeiptr,
                    mvp.modelview.as_ptr() as *const GLvoid,
                    gl::DYNAMIC_DRAW,
                );
            }

            need_render = true;
            need_camera_update = false;
        }

        // find face under mouse cursor
        if need_selection_update {
            // get the raycast result, testing previous selection first
            let raycast = selection.as_ref().and_then(|selection| match *selection {
                Selection::Face(face_id) => {
                    let triangle = mesh.get_face_points(mesh[face_id]);
                    raycast_triangle(triangle, mouse_ray).map(|pos| (pos, face_id))
                },
                Selection::Vertex(_, ref face_ids) => {
                    let mut raycast = None;
                    for &face_id in face_ids.iter() {
                        let triangle = mesh.get_face_points(mesh[face_id]);
                        raycast = raycast_triangle(triangle, mouse_ray)
                            .map(|pos| (pos, face_id));
                        if raycast.is_some() { break; }
                    }
                    raycast
                },
                Selection::Edge(_, ref face_ids) => {
                    let mut raycast = None;
                    for &face_id in face_ids.iter() {
                        let triangle = mesh.get_face_points(mesh[face_id]);
                        raycast = raycast_triangle(triangle, mouse_ray)
                            .map(|pos| (pos, face_id));
                        if raycast.is_some() { break; }
                    }
                    raycast
                },
            }).or_else(|| raycast_mesh(&mesh, mouse_ray));

            // convert raycast into selection
            let new_selection = raycast.map(|(pos, face_id)| {
                let face = mesh[face_id];
                let bary = barycentric(mesh.get_face_points(face), pos);

                const VERTEX_SELECT_THRESHOLD: f32 = 0.9;
                const EDGE_SELECT_THRESHOLD:   f32 = 0.05;

                (            // first check vertex seleciton
                    {   if      bary.x >= VERTEX_SELECT_THRESHOLD { Some(face.0) }
                        else if bary.y >= VERTEX_SELECT_THRESHOLD { Some(face.1) }
                        else if bary.z >= VERTEX_SELECT_THRESHOLD { Some(face.2) }
                        else                                      { None         }
                    }.map(
                        |vertex_id| Selection::Vertex(vertex_id, Default::default())
                    )
                ).or_else(|| // if not, check for edge selection
                    {   if      bary.x <= EDGE_SELECT_THRESHOLD { Some(Edge(face.1, face.2)) }
                        else if bary.y <= EDGE_SELECT_THRESHOLD { Some(Edge(face.2, face.0)) }
                        else if bary.z <= EDGE_SELECT_THRESHOLD { Some(Edge(face.0, face.1)) }
                        else                                    { None                       }
                    }.map(
                        |edge|      Selection::Edge(edge, Default::default())
                    )
                ).unwrap_or(  // otherwise select the whole face
                                    Selection::Face(face_id)
                )
            });

            // update selection
            if new_selection != selection {
                selection = new_selection;
                selection_primary.clear();
                selection_secondary.clear();

                // finalise specific selection variants, update render buffer
                if let Some(ref mut selection) = selection {
                    match *selection {
                        Selection::Face(face_id) => {
                            selection_secondary.extend_from_slice(mesh[face_id].as_ref());
                            selection_draw_mode = gl::TRIANGLES;
                        },
                        Selection::Vertex(vertex_id, ref mut face_ids) => {
                            mesh.get_vertex_face_ids(vertex_id, face_ids);
                            for &face_id in face_ids.iter() {
                                let opp = mesh[face_id].edge_opposite(vertex_id);
                                selection_primary.extend_from_slice(&[
                                    vertex_id, opp.0, vertex_id, opp.1
                                ]);
                                selection_secondary.extend_from_slice(opp.as_ref());
                            }
                            selection_draw_mode = gl::LINES;
                        },
                        Selection::Edge(edge, ref mut face_ids) => {
                            *face_ids = mesh.get_edge_face_ids(edge);
                            for &face_id in face_ids.iter() {
                                let opp_edges = mesh[face_id].edges_opposite(edge);
                                selection_secondary.extend_from_slice(opp_edges[0].as_ref());
                                selection_secondary.extend_from_slice(opp_edges[1].as_ref());
                            }
                            selection_primary.extend_from_slice(edge.as_ref());
                            selection_draw_mode = gl::LINES;
                        },
                    }
                }
                need_render = true;
            }
            need_selection_update = false;
        }

        // render scene
        if need_render { unsafe {
            gl::ClearColor(0.9, 0.7, 0.7, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            // render mesh faces
            gl::Enable(gl::CULL_FACE);
            gl::PolygonMode(gl::FRONT, gl::FILL);

            gl::BindVertexArray(mesh_vao);
            gl::EnableVertexAttribArray(a_color);
            gl::DrawElements(
                gl::TRIANGLES, mesh.faces.len() as GLsizei * 3, gl::UNSIGNED_BYTE, ptr::null()
            );

            gl::Enable(gl::BLEND);
            if selection.is_none() { gl::Disable(gl::CULL_FACE); }
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);

            let unselected_color = Vector4::new(1.0, 1.0, 1.0, 0.1);
            gl::DisableVertexAttribArray(a_color);
            gl::VertexAttrib4fv(a_color, unselected_color.as_ptr());
            gl::DrawElements(
                gl::TRIANGLES, mesh.faces.len() as GLsizei * 3, gl::UNSIGNED_BYTE, ptr::null()
            );

            if selection.is_some() {
                gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);

                let selected_color = Vector4::new(1.0, 1.0, 1.0, 0.4);
                gl::VertexAttrib4fv(a_color, selected_color.as_ptr());
                gl::DrawElements(
                    selection_draw_mode, selection_secondary.len() as GLsizei,
                    gl::UNSIGNED_BYTE, selection_secondary.as_ptr() as *const GLvoid,
                );

                let selected_color = Vector4::new(1.0, 1.0, 1.0, 1.0);
                gl::VertexAttrib4fv(a_color, selected_color.as_ptr());
                gl::DrawElements(
                    selection_draw_mode, selection_primary.len() as GLsizei,
                    gl::UNSIGNED_BYTE, selection_primary.as_ptr() as *const GLvoid,
                );

                gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, mesh_ebo);
            }
            gl::Disable(gl::BLEND);

            gl_window.swap_buffers().expect("buffer swap failed");
            need_render = false;
        }}
        std::thread::sleep_ms(15); // TODO: calculate smarter sleep timing
    }
}

fn barycentric(t: [Point3<f32>; 3], p: Point3<f32>) -> Vector3<f32> {
    // algorithm from Real-Time Collision Detection, Christer Ericson
    let v0 = t[1] - t[0];
    let v1 = t[2] - t[0];
    let v2 = p    - t[0];

    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);

    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    Vector3::new(u, v, w)
}

#[derive(Copy, Clone, Debug)]
struct Plane {
    normal: Vector3<f32>,
    offset: f32,
}

#[derive(Copy, Clone, Debug)]
struct Ray {
    origin: Point3<f32>,
    direction: Vector3<f32>,
}

impl Ray {
    #[inline]
    fn eval(&self, t: f32) -> Point3<f32> {
        self.origin + (t * self.direction)
    }
}

fn raycast_plane(plane: Plane, ray: Ray) -> Option<Point3<f32>> {
    let plane_ray_dot = plane.normal.dot(ray.direction);
    if plane_ray_dot >= 0.0 { return None; } // fail if ray parallel or towards back face

    let t = (plane.offset - plane.normal.dot(ray.origin.to_vec())) / plane_ray_dot;
    if t < 0.0 { return None } // fail if plane behind ray origin

    Some(ray.eval(t))
}

fn raycast_triangle(triangle: [Point3<f32>; 3], ray: Ray) -> Option<Point3<f32>> {
    let normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]).normalize();
    let offset = normal.dot(triangle[0].to_vec());

    raycast_plane(Plane { normal, offset }, ray).and_then(|point| {
        for i in 0..3 {
            if normal.dot((triangle[(i+1)%3] - triangle[i]).cross(point - triangle[i])) < 0.0 {
                return None; // fail if point lies on right hand side of any directed edge
            }
        }
        Some(point)
    })
}

fn raycast_mesh(mesh: &Mesh, ray: Ray) -> Option<(Point3<f32>, FaceIndex)> {
    mesh.faces.iter()
        .map(|face| mesh.get_face_points(*face))
        .enumerate()
        .filter_map(|(index, triangle)| {
            raycast_triangle(triangle, ray).map(|point| (point, FaceIndex(index as u8)))
        })
        .nth(0) // ASSUMPTION: no mesh faces lie in front or behind another
}

fn make_icosahedron() -> ([Vertex; 12], [Face; 20]) {
    fn face(v0: usize, v1: usize, v2: usize) -> Face {
        Face(VertexIndex(v0 as u8), VertexIndex(v1 as u8), VertexIndex(v2 as u8))
    }

    // construct icosahedron as a gyroelongated bipyramid
    let mut verts = <[Vertex; 12]>::default();

    // set vertex at apex on unit sphere
    verts[5]  = Vertex::at(Point3::new(0.0, 0.0, 1.0));
    verts[11] = Vertex::at(Point3::new(0.0, 0.0, -1.0));

    // create verts at base of top pyramid
    for i in 0..5 {
        let long = Rad(-f32::atan(1.0 / 2.0));
        let lat = Deg(i as f32 * 72.0);

        let rotation = Quaternion::from_angle_z(lat)
                     * Quaternion::from_angle_y(long);

        verts[i] = Vertex::at(rotation.rotate_point(Point3::new(1.0, 0.0, 0.0)));
    }

    // create verts of bottom pyramid by reflecting along z and rotating 1/5 pi about z
    for i in 0..5 {
        let twist = Quaternion::from_angle_z(Deg(36.0));
        let v = verts[i].pos;
        verts[6 + i] = Vertex::at(twist.rotate_point(Point3::new(v.x, v.y, -v.z)));
    }

    let mut faces = <[Face; 20]>::default();
    // assemble top and bottom pyramid faces
    for i in 0..5 {
        faces[i] = face(i, 5, (i+1)%5);
        faces[19-i] = face(6+(i+1)%5, 11, 6+i);
    }
    // assemble connecting faces between pyramids
    for i in 0..5 {
        faces[5+i] = face(i, (i+1)%5, 6+i);
        faces[14-i] = face(6+(i+1)%5, 6+i, (i+6)%5);
    }

    (verts, faces)
}

#[inline]
unsafe fn gen_object(gl_gen_callback: unsafe fn (GLsizei, *mut GLuint)) -> GLuint {
    let mut name = GLuint::default();
    gl_gen_callback(1, &mut name);
    name
}

fn compile_shader(src: &str, ty: GLenum) -> GLuint {
    unsafe {
        let shader = gl::CreateShader(ty);
        gl::ShaderSource(
            shader, 1,
            &(src.as_ptr() as *const GLchar) as *const *const _,
            &(src.len() as GLint) as *const _
        );
        gl::CompileShader(shader);

        // check shader compile errors
        let mut status = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

        if status != (gl::TRUE as GLint) {
            use std::str::from_utf8;

            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::with_capacity(len as usize);
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetShaderInfoLog(
                shader,
                len,
                ptr::null_mut(),
                buf.as_mut_ptr() as *mut GLchar,
            );
            panic!(
                "GLSL compile error:\n{}",
                from_utf8(&buf).ok().expect("ShaderInfoLog not valid utf8")
            );
        }
        shader
    }
}

fn link_shaders(shaders: &[GLuint]) -> GLuint {
    unsafe {
        let program = gl::CreateProgram();
        for &shader in shaders { gl::AttachShader(program, shader); }
        gl::LinkProgram(program);

        {   // check link status
            let mut status = gl::FALSE as GLint;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);
            if status != (gl::TRUE as GLint) {
                use std::str::from_utf8;

                let mut len = 0;
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
                let mut buf = Vec::with_capacity(len as usize);
                buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
                gl::GetProgramInfoLog(
                    program,
                    len,
                    ptr::null_mut(),
                    buf.as_mut_ptr() as *mut GLchar,
                );
                panic!(
                    "GLSL link error:\n{}",
                    from_utf8(&buf).ok().expect("ProgramInfoLog not valid utf8")
                );
            }
        }
        program
    }
}
