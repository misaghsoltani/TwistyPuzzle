//! Trackball camera control: drag to spin the puzzle, let go and it keeps
//! spinning.
//!
//! Rotation only. Panning would let the puzzle leave the frame and zooming is
//! the camera distance's job, so neither is offered. The damping is what makes
//! a flick feel like a flick rather than a jump, and it is not decoration.
//!
//! Derived from the `TrackballControls` of the three.js examples (MIT
//! license).

use crate::math::{Quat, Vec3};

use super::camera::Camera;

/// Pointer coordinates normalized to `[-1, 1]`, with `y` up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pointer {
    pub x: f64,
    pub y: f64,
}

/// `rotateSpeed = 3`, pan and zoom disabled.
#[derive(Clone, Copy, Debug)]
pub struct TrackballControls {
    pub rotate_speed: f64,
    pub dynamic_damping_factor: f64,
    pub static_moving: bool,
    pub enabled: bool,
    target: Vec3,
    eye: Vec3,
    move_prev: Vec3,
    move_curr: Vec3,
    last_axis: Vec3,
    last_angle: f64,
    dragging: bool,
}

impl Default for TrackballControls {
    fn default() -> Self {
        TrackballControls {
            rotate_speed: 3.0,
            dynamic_damping_factor: 0.2,
            static_moving: false,
            enabled: true,
            target: Vec3::default(),
            eye: Vec3::default(),
            move_prev: Vec3::default(),
            move_curr: Vec3::default(),
            last_axis: Vec3::new(0.0, 1.0, 0.0),
            last_angle: 0.0,
            dragging: false,
        }
    }
}

impl TrackballControls {
    pub fn new() -> TrackballControls {
        TrackballControls::default()
    }

    /// Map the pointer onto a unit circle centered on the viewport.
    fn on_circle(p: Pointer) -> Vec3 {
        Vec3::new(p.x, p.y, 0.0)
    }

    /// Begin a drag.
    pub fn pointer_down(&mut self, p: Pointer) {
        if !self.enabled {
            return;
        }
        self.move_curr = Self::on_circle(p);
        self.move_prev = self.move_curr;
        self.dragging = true;
    }

    /// Continue a drag.
    pub fn pointer_move(&mut self, p: Pointer) {
        if !self.enabled || !self.dragging {
            return;
        }
        self.move_prev = self.move_curr;
        self.move_curr = Self::on_circle(p);
    }

    pub fn pointer_up(&mut self) {
        self.dragging = false;
    }

    pub fn is_dragging(&self) -> bool {
        self.dragging
    }

    /// `rotateCamera`: rotate the eye about an axis perpendicular to the drag.
    ///
    /// Returns `true` when the camera moved, which is the caller's cue to
    /// request another frame.
    pub fn update(&mut self, camera: &mut Camera) -> bool {
        if !self.enabled {
            return false;
        }
        self.eye = camera.position.sub(&self.target);

        let mut moved = false;
        let delta = self.move_curr.sub(&self.move_prev);
        let mut angle = delta.length();

        if angle > 0.0 {
            let eye_dir = self.eye.normalize();
            let up_dir = camera.up.normalize();
            let side_dir = up_dir.cross(&eye_dir).normalize();

            let side = side_dir.scale(self.move_curr.x - self.move_prev.x);
            let up = up_dir.scale(self.move_curr.y - self.move_prev.y);
            let move_dir = side.add(&up);
            let axis = move_dir.cross(&eye_dir).normalize();

            angle *= self.rotate_speed;
            let q = Quat::from_axis_angle(&axis, angle);
            self.eye = self.eye.apply_quat(&q);
            camera.up = camera.up.apply_quat(&q);

            self.last_axis = axis;
            self.last_angle = angle;
            moved = true;
        } else if !self.static_moving && self.last_angle != 0.0 {
            // Damped spin-down after the pointer stops.
            self.last_angle *= (1.0 - self.dynamic_damping_factor).sqrt();
            let q = Quat::from_axis_angle(&self.last_axis, self.last_angle);
            self.eye = self.eye.apply_quat(&q);
            camera.up = camera.up.apply_quat(&q);
            moved = self.last_angle.abs() > 1e-6;
        }

        if self.static_moving {
            self.move_prev = self.move_curr;
        } else {
            let d = self
                .move_curr
                .sub(&self.move_prev)
                .scale(self.dynamic_damping_factor);
            self.move_prev = self.move_prev.add(&d);
        }

        camera.position = self.target.add(&self.eye);
        camera.target = self.target;
        moved
    }

    /// Restore the initial orientation.
    pub fn reset(&mut self, camera: &mut Camera) {
        let dist = camera.position.sub(&self.target).length();
        camera.position = self.target.add(&Vec3::new(0.0, 0.0, dist));
        camera.up = Vec3::new(0.0, 1.0, 0.0);
        self.move_prev = Vec3::default();
        self.move_curr = Vec3::default();
        self.last_angle = 0.0;
        self.dragging = false;
    }
}
