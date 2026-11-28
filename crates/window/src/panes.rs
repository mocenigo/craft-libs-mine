//! A window drawn by two egui contexts: the app's own and a pane with its own zoom.
//!
//! egui keeps one zoom per [`egui::Context`], shared by every viewport, so a strip that must keep the platform's
//! size (a title bar drawn by the app) cannot stay put while the rest of the window zooms. An app that opts in
//! makes a [`Pane`]: a second context, run inside the window's pass and drawn into the same surface. The pane
//! lies over the whole window and is transparent where it draws nothing; it takes the pointer over its strip
//! ([`Pane::show`]'s `strip`) and over its own popups, and the window's context takes everything else.
//!
//! Each pass of the window's context:
//!
//! - before input is read, the pane takes its events: the pointer where it lies (and through a drag that began
//!   on it), the keys while it holds keyboard focus (shortcuts with the command key stay with the window), and
//!   the accessibility actions aimed at its nodes. Tab moves keyboard focus between the two contexts;
//! - [`Pane::show`] runs the pane's UI at its zoom, tessellates it at its own pixels per point, so text is
//!   rasterized for the size it has on screen, and hands its textures to the window's texture manager;
//! - after the window's pass, the pane's meshes are drawn over the window's, and its cursor, IME area, output
//!   commands, window commands and repaint requests are merged in. Its accessibility tree becomes a subtree of
//!   the window's, with its own node ids and a transform from its points to the window's.
//!
//! Apps that make no pane are not affected.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use egui::accesskit::{self, NodeId};
use egui::epaint::{ClippedShape, Primitive};
use egui::{Context, Event, FullOutput, Id, Key, Modifiers, Order, Pos2, RawInput, Rect, Shape, TextureId, Ui, ViewportId};

/// Which context an input goes to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Owner {
    #[default]
    Window,
    Pane,
}

/// State shared between the [`Pane`] and the plugin on the window's context.
#[derive(Default)]
struct Shared {
    /// Salts the pane's ids in the window (accessibility nodes, the IME owner).
    salt: Option<Id>,
    /// The pane's pixels per point at its last run.
    pane_ppp: f32,
    /// Where the pane always takes the pointer, in pane points.
    strip: Option<Rect>,
    /// The pane's top left corner, in window points.
    origin: Pos2,
    /// Events for the pane's next run.
    events: Vec<Event>,
    hover: Owner,
    /// The context a pressed button belongs to until every button is released.
    capture: Option<Owner>,
    keyboard: Owner,
    /// The first and last widget of each context's Tab order at its last pass.
    pane_tab: Option<(Id, Id)>,
    window_tab: Option<(Id, Id)>,
    /// Pane texture → window texture.
    textures: HashMap<TextureId, TextureId>,
    /// Window node → pane node, for the nodes of the last merged tree.
    nodes: HashMap<NodeId, NodeId>,
    /// What the pane's last run leaves for the window's output.
    out: Option<PaneOutput>,
}

/// The pane's last run, in window points.
struct PaneOutput {
    shapes: Vec<ClippedShape>,
    platform: egui::PlatformOutput,
    viewport: Option<egui::ViewportOutput>,
    /// Pane points → window points: `origin + p * scale`.
    origin: Pos2,
    scale: f32,
    /// The pane's strip on screen, in pane points: its node's bounds.
    bounds: Rect,
}

impl Shared {
    fn to_pane(&self, ctx: &Context, pos: Pos2) -> Pos2 {
        let window_ppp = ctx.native_pixels_per_point().unwrap_or(1.0) * ctx.zoom_factor();
        let scale = self.pane_ppp / window_ppp;
        Pos2::ZERO + (pos - self.origin) / scale
    }

    /// Whether the pane takes the pointer at `pos` (pane points): its strip, or one of its areas (a popup).
    fn wants(&self, pane: &Context, pos: Pos2) -> bool {
        self.strip.is_some_and(|s| s.contains(pos)) || pane.layer_id_at(pos).is_some_and(|layer| layer.order != Order::Background)
    }
}

/// A part of the window drawn by its own context at its own zoom. See the [module docs](self).
pub struct Pane {
    ctx: Context,
    shared: Arc<Mutex<Shared>>,
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Pane {
    /// Makes a pane for the window whose context is `window`, named `name` (unique in the window).
    ///
    /// shortcut: one pane per window (egui keeps one plugin of a type per context); a list in [`Shared`] when
    /// an app needs more.
    pub fn new(window: &Context, name: &str) -> Self {
        let ctx = Context::default();
        // The pane's zoom is the app's choice, made in `show`; egui's own zoom keys would fight it.
        ctx.options_mut(|o| {
            o.zoom_with_keyboard = false;
            o.quit_shortcuts.clear();
        });
        ctx.enable_accesskit();
        let host = window.clone();
        ctx.set_request_repaint_callback(move |info| host.request_repaint_after(info.delay));
        let shared = Arc::new(Mutex::new(Shared { salt: Some(Id::unique(("craft-window pane", name))), pane_ppp: 1.0, ..Default::default() }));
        window.add_plugin(Plugin { pane: ctx.clone(), shared: shared.clone() });
        Self { ctx, shared }
    }

    /// The pane's own context, to give it the app's fonts, style and theme.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Runs the pane's UI for this pass of `window` at `zoom` over the window's native scale, and draws it over
    /// the window when the pass ends. The pane's top left corner is at `origin` (window points) and it reaches the
    /// window's bottom right corner; `strip` (pane points) is where it always takes the pointer.
    /// Call it in every pass of the window.
    pub fn show(&self, window: &Context, zoom: f32, origin: Pos2, strip: Rect, add_contents: impl FnOnce(&mut Ui)) {
        let native = window.native_pixels_per_point().unwrap_or(1.0);
        let window_ppp = window.pixels_per_point();
        let pane_ppp = native * zoom;
        let scale = pane_ppp / window_ppp;
        let mut info = window.input(|i| i.viewport().clone());
        info.native_pixels_per_point = Some(native);
        let mut raw = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, (window.viewport_rect().max - origin) / scale)),
            time: window.input(|i| Some(i.time)),
            max_texture_side: window.input(|i| Some(i.max_texture_side)),
            focused: window.input(|i| i.focused),
            system_theme: window.system_theme(),
            ..Default::default()
        };
        raw.viewports.insert(ViewportId::ROOT, info);
        let events;
        {
            let mut shared = lock(&self.shared);
            shared.pane_ppp = pane_ppp;
            shared.strip = Some(strip);
            shared.origin = origin;
            events = std::mem::take(&mut shared.events);
        }
        raw.events = events;
        if self.ctx.zoom_factor() != zoom {
            self.ctx.set_zoom_factor(zoom);
        }
        let mut add_contents = Some(add_contents);
        let out = self.ctx.run_ui(raw, |ui| {
            if let Some(add) = add_contents.take() {
                add(ui);
            }
        });
        let tab = self.ctx.memory(|m| tab_ends(m.focus_candidates()));
        let mut shared = lock(&self.shared);
        shared.pane_tab = tab;
        let shapes = self.composite(window, &mut shared, out.textures_delta, out.shapes, out.pixels_per_point, origin, scale);
        let mut viewport_output = out.viewport_output;
        let bounds = strip.intersect(Rect::from_min_size(Pos2::ZERO, (window.viewport_rect().max - origin) / scale));
        shared.out =
            Some(PaneOutput { shapes, platform: out.platform_output, viewport: viewport_output.remove(&ViewportId::ROOT), origin, scale, bounds });
    }

    /// Hands the pane's textures to the window and turns its shapes into meshes in window points.
    #[expect(clippy::too_many_arguments)]
    fn composite(
        &self,
        window: &Context,
        shared: &mut Shared,
        mut textures: egui::TexturesDelta,
        shapes: Vec<ClippedShape>,
        pane_ppp: f32,
        origin: Pos2,
        scale: f32,
    ) -> Vec<ClippedShape> {
        let manager = window.tex_manager();
        let mut manager = manager.write();
        for (id, deltas) in textures.set.drain() {
            for delta in deltas {
                match shared.textures.get(&id) {
                    Some(&mapped) => manager.set(mapped, delta),
                    None if delta.pos.is_none() => {
                        let mapped = manager.alloc(format!("pane {id:?}"), delta.image, delta.options);
                        shared.textures.insert(id, mapped);
                    }
                    None => {}
                }
            }
        }
        for id in std::mem::take(&mut textures.free) {
            if let Some(mapped) = shared.textures.remove(&id) {
                manager.free(mapped);
            }
        }
        drop(manager);
        let to_window = |p: Pos2| origin + p.to_vec2() * scale;
        let rect = |r: Rect| Rect::from_min_max(to_window(r.min), to_window(r.max));
        self.ctx
            .tessellate(shapes, pane_ppp)
            .into_iter()
            .filter_map(|clipped| {
                // shortcut: paint callbacks are dropped; a pane that draws with the GPU needs them forwarded.
                let Primitive::Mesh(mut mesh) = clipped.primitive else { return None };
                mesh.texture_id = *shared.textures.get(&mesh.texture_id)?;
                for v in &mut mesh.vertices {
                    v.pos = to_window(v.pos);
                }
                Some(ClippedShape {
                    clip_rect: rect(clipped.clip_rect),
                    shape: Shape::mesh(mesh),
                    transform_after_tessellation: egui::emath::TSTransform::IDENTITY,
                })
            })
            .collect()
    }
}

/// The first and last ids of a Tab order.
fn tab_ends(candidates: &[egui::FocusCandidate]) -> Option<(Id, Id)> {
    Some((candidates.first()?.id, candidates.last()?.id))
}

fn surrender(ctx: &Context) {
    ctx.memory_mut(|m| {
        if let Some(id) = m.focused() {
            m.surrender_focus(id);
        }
    });
}

/// The window context's side of the pane: routes input before the pass and merges the pane's output after it.
struct Plugin {
    pane: Context,
    shared: Arc<Mutex<Shared>>,
}

impl Plugin {
    /// Gives keyboard focus to `to`, on the first or last widget of its Tab order.
    fn move_focus(&self, window: &Context, shared: &mut Shared, to: Owner, first: bool) {
        let (from_ctx, to_ctx, ends) = match to {
            Owner::Pane => (window, &self.pane, shared.pane_tab),
            Owner::Window => (&self.pane, window, shared.window_tab),
        };
        surrender(from_ctx);
        if let Some((a, b)) = ends {
            to_ctx.memory_mut(|m| m.request_focus(if first { a } else { b }));
        }
        shared.keyboard = to;
    }

    /// Where a Tab press goes: `None` when it moves focus from one context to the other.
    fn tab(&self, window: &Context, shared: &mut Shared, back: bool) -> Option<Owner> {
        let (ctx, ends, other, other_ends) = match shared.keyboard {
            Owner::Pane => (&self.pane, shared.pane_tab, Owner::Window, shared.window_tab),
            Owner::Window => (window, shared.window_tab, Owner::Pane, shared.pane_tab),
        };
        let focused = ctx.memory(|m| m.focused());
        let at_end = match ends {
            None => true,
            Some((first, last)) => focused.is_some() && focused == Some(if back { first } else { last }),
        };
        if at_end && other_ends.is_some() && (shared.keyboard == Owner::Pane || focused.is_some()) {
            self.move_focus(window, shared, other, !back);
            return None;
        }
        Some(shared.keyboard)
    }
}

impl egui::plugin::Plugin for Plugin {
    fn debug_name(&self) -> &'static str {
        "craft-window pane"
    }

    fn input_hook(&mut self, ctx: &Context, input: &mut RawInput) {
        let mut shared = lock(&self.shared);
        let mut window_events = Vec::with_capacity(input.events.len());
        for event in input.events.drain(..) {
            let to = match &event {
                Event::PointerMoved(pos) => {
                    let p = shared.to_pane(ctx, *pos);
                    let to = shared.capture.unwrap_or(if shared.wants(&self.pane, p) { Owner::Pane } else { Owner::Window });
                    if to != shared.hover {
                        match to {
                            Owner::Pane => window_events.push(Event::PointerGone),
                            Owner::Window => shared.events.push(Event::PointerGone),
                        }
                        shared.hover = to;
                    }
                    if to == Owner::Pane {
                        shared.events.push(Event::PointerMoved(p));
                        continue;
                    }
                    Owner::Window
                }
                Event::PointerButton { pos, button, pressed, modifiers } => {
                    let p = shared.to_pane(ctx, *pos);
                    // A press goes where it is; a release, to where the press went.
                    let here = if shared.wants(&self.pane, p) { Owner::Pane } else { Owner::Window };
                    let to = if *pressed { here } else { shared.capture.unwrap_or(here) };
                    if *pressed {
                        if shared.capture.is_none() && shared.keyboard != to {
                            surrender(if to == Owner::Pane { ctx } else { &self.pane });
                            shared.keyboard = to;
                        }
                        shared.capture = Some(to);
                    } else {
                        // shortcut: one button at a time; a chord released in another order ends the capture early.
                        shared.capture = None;
                    }
                    if to == Owner::Pane {
                        shared.events.push(Event::PointerButton { pos: p, button: *button, pressed: *pressed, modifiers: *modifiers });
                        continue;
                    }
                    Owner::Window
                }
                Event::MouseWheel { .. } => shared.hover,
                Event::PointerGone => {
                    shared.capture = None;
                    shared.events.push(Event::PointerGone);
                    Owner::Window
                }
                Event::WindowFocused(_) | Event::SystemTheme(_) => {
                    shared.events.push(event.clone());
                    Owner::Window
                }
                Event::Key { key: Key::Tab, pressed, modifiers, .. } if (modifiers.is_none() || *modifiers == Modifiers::SHIFT) => {
                    if *pressed {
                        match self.tab(ctx, &mut shared, modifiers.shift) {
                            Some(to) => to,
                            None => continue,
                        }
                    } else {
                        shared.keyboard
                    }
                }
                // The window's shortcuts work wherever the focus is.
                Event::Key { modifiers, .. } if modifiers.command => Owner::Window,
                Event::Key { .. } | Event::Text(_) | Event::Copy | Event::Cut | Event::Paste(_) | Event::Ime(_) => shared.keyboard,
                Event::AccessKitActionRequest(request) => match shared.nodes.get(&request.target_node) {
                    Some(&node) => {
                        let mut request = request.clone();
                        request.target_node = node;
                        if request.action == accesskit::Action::Focus {
                            surrender(ctx);
                            shared.keyboard = Owner::Pane;
                        }
                        shared.events.push(Event::AccessKitActionRequest(request));
                        continue;
                    }
                    None => Owner::Window,
                },
                // shortcut: touches, pinches and drops stay with the window; a pane that scrolls by touch needs them.
                _ => Owner::Window,
            };
            match to {
                Owner::Window => window_events.push(event),
                Owner::Pane => {
                    let event = match event {
                        Event::MouseWheel { unit, delta, phase, source, modifiers } => {
                            let window_ppp = ctx.native_pixels_per_point().unwrap_or(1.0) * ctx.zoom_factor();
                            let delta = if unit == egui::MouseWheelUnit::Point { delta * window_ppp / shared.pane_ppp } else { delta };
                            Event::MouseWheel { unit, delta, phase, source, modifiers }
                        }
                        other => other,
                    };
                    shared.events.push(event);
                }
            }
        }
        input.events = window_events;
    }

    fn output_hook(&mut self, ctx: &Context, output: &mut FullOutput) {
        let mut shared = lock(&self.shared);
        shared.window_tab = ctx.memory(|m| tab_ends(m.focus_candidates()));
        let Some(pane) = shared.out.take() else { return };
        let salt = shared.salt.unwrap_or(Id::NULL);
        output.shapes.extend(pane.shapes);
        let to_window = |r: Rect| Rect::from_min_max(pane.origin + r.min.to_vec2() * pane.scale, pane.origin + r.max.to_vec2() * pane.scale);
        let platform = &mut output.platform_output;
        if shared.hover == Owner::Pane || shared.capture == Some(Owner::Pane) {
            platform.cursor_icon = pane.platform.cursor_icon;
        }
        if shared.keyboard == Owner::Pane
            && let Some(mut ime) = pane.platform.ime
        {
            ime.rect = to_window(ime.rect);
            ime.cursor_rect = to_window(ime.cursor_rect);
            ime.owner = salt.with(ime.owner);
            platform.ime = Some(ime);
        }
        platform.commands.extend(pane.platform.commands);
        if let Some(viewport) = pane.viewport
            && let Some(root) = output.viewport_output.get_mut(&ViewportId::ROOT)
        {
            // The system takes the pointer for a window drag and may never report the release.
            if viewport.commands.iter().any(|c| matches!(c, egui::ViewportCommand::StartDrag)) {
                shared.capture = None;
                shared.events.push(Event::PointerGone);
            }
            root.commands.extend(viewport.commands);
            root.repaint_delay = root.repaint_delay.min(viewport.repaint_delay);
        }
        if let (Some(window_tree), Some(pane_tree)) = (platform.accesskit_update.as_mut(), pane.platform.accesskit_update) {
            shared.nodes = merge_tree(window_tree, pane_tree, salt, (pane.origin, pane.scale, pane.bounds), shared.keyboard == Owner::Pane);
        }
    }
}

/// Puts `pane`'s nodes under the root of `window`, first among its children, and returns window node → pane node.
fn merge_tree(
    window: &mut accesskit::TreeUpdate,
    pane: accesskit::TreeUpdate,
    salt: Id,
    place: (Pos2, f32, Rect),
    focused: bool,
) -> HashMap<NodeId, NodeId> {
    let (origin, scale, b) = place;
    let map = |n: NodeId| NodeId(salt.with(n.0).value());
    let Some(pane_root) = pane.tree.map(|t| t.root) else { return HashMap::new() };
    let mut back = HashMap::with_capacity(pane.nodes.len());
    for (id, mut node) in pane.nodes {
        let children: Vec<NodeId> = node.children().iter().copied().map(map).collect();
        node.set_children(children);
        // shortcut: only `children` and `labelled_by` (flattened by egui) are remapped; other relations between
        // nodes (`controls`, `active_descendant`…) still name pane ids.
        if id == pane_root {
            node.set_role(accesskit::Role::Pane);
            node.set_bounds(accesskit::Rect::new(f64::from(b.min.x), f64::from(b.min.y), f64::from(b.max.x), f64::from(b.max.y)));
            node.set_transform(accesskit::Affine::translate((f64::from(origin.x), f64::from(origin.y))) * accesskit::Affine::scale(f64::from(scale)));
        }
        back.insert(map(id), id);
        window.nodes.push((map(id), node));
    }
    let window_root = window.tree.as_ref().map(|t| t.root);
    if let Some((_, root)) = window.nodes.iter_mut().find(|(id, _)| Some(*id) == window_root) {
        let mut children = vec![map(pane_root)];
        children.extend_from_slice(root.children());
        root.set_children(children);
    }
    if focused && pane.focus != pane_root {
        window.focus = map(pane.focus);
    }
    back
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Vec2;

    /// A window at `zoom` with a pane at zoom 1 whose strip holds a "Bar" button, and a "Page" button under it.
    struct Window {
        ctx: Context,
        pane: Pane,
        clicked: Vec<&'static str>,
    }

    impl Window {
        fn new(zoom: f32) -> Self {
            let ctx = Context::default();
            ctx.enable_accesskit();
            ctx.set_zoom_factor(zoom);
            let pane = Pane::new(&ctx, "bar");
            Self { ctx, pane, clicked: Vec::new() }
        }

        fn frame(&mut self, events: Vec<Event>) -> FullOutput {
            let raw = RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0) / self.ctx.zoom_factor())),
                events,
                ..Default::default()
            };
            let (pane, clicked) = (&self.pane, &mut self.clicked);
            let mut out = self.ctx.run_ui(raw, |ui| {
                let strip = Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 40.0));
                pane.show(ui.ctx(), 1.0, ui.max_rect().min, strip, |ui| {
                    if ui.button("Bar").clicked() {
                        clicked.push("Bar");
                    }
                });
                // The strip's 40 pixels, in window points.
                ui.add_space(40.0 / ui.ctx().pixels_per_point() + 20.0);
                if ui.button("Page").clicked() {
                    clicked.push("Page");
                }
            });
            // No painter here.
            out.textures_delta.clear();
            out
        }

        /// Where the node labelled `label` is, in window points, from the window's accessibility tree.
        fn find(&mut self, label: &str) -> Rect {
            let out = self.frame(Vec::new());
            let tree = out.platform_output.accesskit_update.unwrap_or_else(|| accesskit::TreeUpdate {
                nodes: Vec::new(),
                tree: None,
                tree_id: accesskit::TreeId::ROOT,
                focus: NodeId(0),
            });
            let parent: HashMap<NodeId, NodeId> = tree.nodes.iter().flat_map(|(id, n)| n.children().iter().map(move |c| (*c, *id))).collect();
            let nodes: HashMap<NodeId, &accesskit::Node> = tree.nodes.iter().map(|(id, n)| (*id, n)).collect();
            let Some((mut id, node)) = tree.nodes.iter().find(|(_, n)| n.label() == Some(label)).map(|(id, n)| (*id, n)) else {
                return Rect::NOTHING;
            };
            let b = node.bounds().unwrap_or_default();
            let mut affine = accesskit::Affine::IDENTITY;
            while let Some(&p) = parent.get(&id) {
                // Up to the window's root, whose transform turns points into pixels.
                if parent.contains_key(&p)
                    && let Some(t) = nodes[&p].transform()
                {
                    affine = *t * affine;
                }
                id = p;
            }
            let r = affine.transform_rect_bbox(b);
            Rect::from_min_max(egui::pos2(r.x0 as f32, r.y0 as f32), egui::pos2(r.x1 as f32, r.y1 as f32))
        }

        fn click(&mut self, label: &str) {
            let at = self.find(label).center();
            let press = |pressed| Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
            self.frame(vec![Event::PointerMoved(at)]);
            self.frame(vec![press(true)]);
            self.frame(vec![press(false)]);
            self.frame(Vec::new());
        }

        /// The size in pixels of what the pane draws.
        fn pane_pixels(&mut self) -> Vec2 {
            let out = self.frame(Vec::new());
            let mapped: Vec<TextureId> = lock(&self.pane.shared).textures.values().copied().collect();
            let mut bounds = Rect::NOTHING;
            for shape in &out.shapes {
                if let Shape::Mesh(mesh) = &shape.shape
                    && mapped.contains(&mesh.texture_id)
                {
                    for v in &mesh.vertices {
                        bounds.extend_with(v.pos);
                    }
                }
            }
            bounds.size() * out.pixels_per_point
        }
    }

    #[test]
    fn the_pane_keeps_its_pixel_size_and_its_clicks_when_the_window_zooms() {
        let mut sizes = Vec::new();
        for zoom in [1.0, 2.0, 0.5] {
            let mut w = Window::new(zoom);
            for _ in 0..3 {
                w.frame(Vec::new());
            }
            sizes.push(w.pane_pixels());
            assert!(w.find("Bar").is_positive(), "the pane's button is in the window's tree at zoom {zoom}");
            w.click("Bar");
            w.click("Page");
            assert_eq!(w.clicked, ["Bar", "Page"], "each click lands in its own context at zoom {zoom}");
        }
        assert!(sizes.iter().all(|s| (*s - sizes[0]).length() < 0.5 && s.x > 0.0), "the pane draws at the same pixel size: {sizes:?}");
    }
    #[test]
    fn tab_moves_focus_from_the_window_into_the_pane_and_back() {
        let mut w = Window::new(2.0);
        for _ in 0..3 {
            w.frame(Vec::new());
        }
        let tab = || Event::Key { key: Key::Tab, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE };
        let mut focused = Vec::new();
        for _ in 0..3 {
            w.frame(vec![tab()]);
            let out = w.frame(Vec::new());
            let tree = out.platform_output.accesskit_update.expect("accesskit is on");
            let label = tree.nodes.iter().find(|(id, _)| *id == tree.focus).and_then(|(_, n)| n.label().map(str::to_owned));
            focused.push(label.unwrap_or_default());
        }
        assert_eq!(focused, ["Page", "Bar", "Page"], "Tab goes round both contexts");
    }
}
