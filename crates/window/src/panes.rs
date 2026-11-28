//! A window drawn by several egui contexts: the app's own and panes, each with its own zoom.
//!
//! egui keeps one zoom per [`egui::Context`], shared by every viewport, so a strip that must keep the platform's
//! size (a title bar or a status bar drawn by the app) cannot stay put while the rest of the window zooms. An app
//! that opts in makes a [`Pane`] for each such strip: a context of its own, run inside the window's pass and drawn
//! into the same surface. A pane lies over the window from its origin to the bottom right corner and is
//! transparent where it draws nothing; it takes the pointer over its strip ([`Pane::show`]'s `strip`) and over its
//! own popups, and the window's context takes everything else.
//!
//! Each pass of the window's context:
//!
//! - before input is read, each pane takes its events: the pointer where it lies (and through a press that began
//!   on it), the keys while it holds keyboard focus (shortcuts with the command key stay with the window), and
//!   the accessibility actions aimed at its nodes. Tab moves keyboard focus round the panes placed
//!   [`Place::Before`] the window's content, the window, and the panes placed [`Place::After`] it;
//! - [`Pane::show`] runs a pane's UI at its zoom, tessellates it at its own pixels per point, so text is
//!   rasterized for the size it has on screen, and hands its textures to the window's texture manager;
//! - after the window's pass, the panes' meshes are drawn over the window's, and their cursor, IME area, output
//!   commands, window commands and repaint requests are merged in. Each pane's accessibility tree becomes a
//!   subtree of the window's, in Tab order, with its own node ids and a transform from its points to the window's.
//!
//! Apps that make no pane are not affected.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use egui::accesskit::{self, NodeId};
use egui::epaint::{ClippedShape, Primitive};
use egui::{Context, Event, FullOutput, Id, Key, Modifiers, Order, Pos2, RawInput, Rect, Shape, TextureId, Ui, ViewportId};

/// Where a pane comes in the window's Tab order and accessibility tree: before the window's own content (a title
/// bar) or after it (a status bar).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    /// Before the window's own content.
    Before,
    /// After the window's own content.
    After,
}

/// Which context an input goes to: the window's own, or the pane at that index.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Owner {
    #[default]
    Window,
    Pane(usize),
}

/// One pane, as the window's plugin sees it.
struct Slot {
    ctx: Context,
    /// False once the [`Pane`] is dropped.
    alive: bool,
    place: Place,
    /// Salts the pane's ids in the window (accessibility nodes, the IME owner).
    salt: Id,
    /// The pane's pixels per point at its last run.
    ppp: f32,
    /// Where the pane always takes the pointer, in pane points.
    strip: Option<Rect>,
    /// The pane's top left corner, in window points.
    origin: Pos2,
    /// Events for the pane's next run.
    events: Vec<Event>,
    /// The first and last widget of the pane's Tab order at its last pass.
    tab: Option<(Id, Id)>,
    /// Pane texture → window texture.
    textures: HashMap<TextureId, TextureId>,
    /// Window node → pane node, for the nodes of the last merged tree.
    nodes: HashMap<NodeId, NodeId>,
    /// What the pane's last run leaves for the window's output.
    out: Option<PaneOutput>,
}

/// The panes of one window, shared by the [`Pane`]s and the plugin on the window's context.
#[derive(Default)]
struct Shared {
    slots: Vec<Slot>,
    hover: Owner,
    /// The context a pressed button belongs to until it is released.
    capture: Option<Owner>,
    keyboard: Owner,
    /// The first and last widget of the window's own Tab order at its last pass.
    window_tab: Option<(Id, Id)>,
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

impl Slot {
    fn to_pane(&self, window: &Context, pos: Pos2) -> Pos2 {
        let window_ppp = window.native_pixels_per_point().unwrap_or(1.0) * window.zoom_factor();
        Pos2::ZERO + (pos - self.origin) * window_ppp / self.ppp
    }

    /// Whether the pane takes the pointer at `pos` (pane points): its strip, or one of its areas (a popup).
    fn wants(&self, pos: Pos2) -> bool {
        self.strip.is_some_and(|s| s.contains(pos)) || self.ctx.layer_id_at(pos).is_some_and(|layer| layer.order != Order::Background)
    }
}

impl Shared {
    /// Who takes the pointer at `pos` (window points): the topmost pane that wants it, else the window.
    fn under(&self, window: &Context, pos: Pos2) -> Owner {
        (0..self.slots.len())
            .rev()
            .find(|&i| self.slots[i].alive && self.slots[i].wants(self.slots[i].to_pane(window, pos)))
            .map_or(Owner::Window, Owner::Pane)
    }

    fn ctx(&self, window: &Context, owner: Owner) -> Context {
        match owner {
            Owner::Window => window.clone(),
            Owner::Pane(i) => self.slots[i].ctx.clone(),
        }
    }

    fn tab_ends(&self, owner: Owner) -> Option<(Id, Id)> {
        match owner {
            Owner::Window => self.window_tab,
            Owner::Pane(i) => self.slots[i].tab,
        }
    }

    /// The Tab order of the contexts: the panes placed before the window's content, the window, the panes after.
    fn cycle(&self) -> Vec<Owner> {
        let panes = |place| (0..self.slots.len()).filter(move |&i| self.slots[i].alive && self.slots[i].place == place).map(Owner::Pane);
        panes(Place::Before).chain([Owner::Window]).chain(panes(Place::After)).collect()
    }

    /// Delivers `event` to `owner`'s next pass, or to the window's events.
    fn send(&mut self, owner: Owner, event: Event, window_events: &mut Vec<Event>) {
        match owner {
            Owner::Window => window_events.push(event),
            Owner::Pane(i) => self.slots[i].events.push(event),
        }
    }

    /// Gives keyboard focus to `to`, on the first or last widget of its Tab order.
    fn move_focus(&mut self, window: &Context, to: Owner, first: bool) {
        surrender(&self.ctx(window, self.keyboard));
        if let Some((a, b)) = self.tab_ends(to) {
            self.ctx(window, to).memory_mut(|m| m.request_focus(if first { a } else { b }));
        }
        self.keyboard = to;
    }

    /// Where a Tab press goes: `None` when it moves focus to another context.
    fn tab(&mut self, window: &Context, back: bool) -> Option<Owner> {
        let here = self.keyboard;
        let focused = self.ctx(window, here).memory(|m| m.focused());
        let at_end = match self.tab_ends(here) {
            None => true,
            Some((first, last)) => focused.is_some() && focused == Some(if back { first } else { last }),
        };
        // With nothing focused in the window, egui's own Tab picks its first widget.
        if !at_end || (here == Owner::Window && focused.is_none()) {
            return Some(here);
        }
        let cycle = self.cycle();
        let at = cycle.iter().position(|o| *o == here)?;
        let n = cycle.len();
        let next = (1..n).map(|k| cycle[if back { (at + n - k) % n } else { (at + k) % n }]).find(|o| self.tab_ends(*o).is_some())?;
        self.move_focus(window, next, !back);
        None
    }
}

/// A part of the window drawn by its own context at its own zoom. See the [module docs](self).
pub struct Pane {
    ctx: Context,
    shared: Arc<Mutex<Shared>>,
    index: usize,
}

fn lock(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Pane {
    /// Makes a pane for the window whose context is `window`, named `name` (unique in the window), coming at
    /// `place` in the window's Tab order and accessibility tree. A window can have several panes; one made later
    /// lies over one made earlier.
    pub fn new(window: &Context, name: &str, place: Place) -> Self {
        let ctx = Context::default();
        // The pane's zoom is the app's choice, made in `show`; egui's own zoom keys would fight it.
        ctx.options_mut(|o| {
            o.zoom_with_keyboard = false;
            o.quit_shortcuts.clear();
        });
        ctx.enable_accesskit();
        let host = window.clone();
        ctx.set_request_repaint_callback(move |info| host.request_repaint_after(info.delay));
        let shared =
            window.data_mut(|d| d.get_temp_mut_or_insert_with(Id::unique("craft-window panes"), || Arc::new(Mutex::new(Shared::default()))).clone());
        // egui keeps one plugin of a type per context: the first pane adds it, and it serves them all.
        window.add_plugin(Plugin { shared: shared.clone() });
        let index = {
            let mut shared = lock(&shared);
            shared.slots.push(Slot {
                ctx: ctx.clone(),
                alive: true,
                place,
                salt: Id::unique(("craft-window pane", name)),
                ppp: 1.0,
                strip: None,
                origin: Pos2::ZERO,
                events: Vec::new(),
                tab: None,
                textures: HashMap::new(),
                nodes: HashMap::new(),
                out: None,
            });
            shared.slots.len() - 1
        };
        Self { ctx, shared, index }
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
        let size = (window.viewport_rect().max - origin) / scale;
        let mut info = window.input(|i| i.viewport().clone());
        info.native_pixels_per_point = Some(native);
        let mut raw = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
            time: window.input(|i| Some(i.time)),
            max_texture_side: window.input(|i| Some(i.max_texture_side)),
            focused: window.input(|i| i.focused),
            system_theme: window.system_theme(),
            ..Default::default()
        };
        raw.viewports.insert(ViewportId::ROOT, info);
        {
            let mut shared = lock(&self.shared);
            let slot = &mut shared.slots[self.index];
            slot.ppp = pane_ppp;
            slot.strip = Some(strip);
            slot.origin = origin;
            raw.events = std::mem::take(&mut slot.events);
        }
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
        let slot = &mut shared.slots[self.index];
        slot.tab = tab;
        let shapes = self.composite(window, slot, out.textures_delta, out.shapes, out.pixels_per_point, origin, scale);
        let mut viewport_output = out.viewport_output;
        let bounds = strip.intersect(Rect::from_min_size(Pos2::ZERO, size));
        slot.out =
            Some(PaneOutput { shapes, platform: out.platform_output, viewport: viewport_output.remove(&ViewportId::ROOT), origin, scale, bounds });
    }

    /// Hands the pane's textures to the window and turns its shapes into meshes in window points.
    #[expect(clippy::too_many_arguments)]
    fn composite(
        &self,
        window: &Context,
        slot: &mut Slot,
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
                match slot.textures.get(&id) {
                    Some(&mapped) => manager.set(mapped, delta),
                    None if delta.pos.is_none() => {
                        let mapped = manager.alloc(format!("pane {id:?}"), delta.image, delta.options);
                        slot.textures.insert(id, mapped);
                    }
                    None => {}
                }
            }
        }
        for id in std::mem::take(&mut textures.free) {
            if let Some(mapped) = slot.textures.remove(&id) {
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
                mesh.texture_id = *slot.textures.get(&mesh.texture_id)?;
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

impl Drop for Pane {
    fn drop(&mut self) {
        let mut shared = lock(&self.shared);
        let index = self.index;
        if let Some(slot) = shared.slots.get_mut(index) {
            slot.alive = false;
            slot.out = None;
        }
        let shared = &mut *shared;
        for owner in [&mut shared.hover, &mut shared.keyboard] {
            if *owner == Owner::Pane(index) {
                *owner = Owner::Window;
            }
        }
        if shared.capture == Some(Owner::Pane(index)) {
            shared.capture = None;
        }
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

/// The window context's side of its panes: routes input before the pass and merges the panes' output after it.
struct Plugin {
    shared: Arc<Mutex<Shared>>,
}

impl egui::plugin::Plugin for Plugin {
    fn debug_name(&self) -> &'static str {
        "craft-window panes"
    }

    fn input_hook(&mut self, ctx: &Context, input: &mut RawInput) {
        let mut shared = lock(&self.shared);
        let mut window_events = Vec::with_capacity(input.events.len());
        for event in input.events.drain(..) {
            let to = match &event {
                Event::PointerMoved(pos) => {
                    let to = shared.capture.unwrap_or_else(|| shared.under(ctx, *pos));
                    let from = shared.hover;
                    if to != from {
                        shared.send(from, Event::PointerGone, &mut window_events);
                        shared.hover = to;
                    }
                    to
                }
                Event::PointerButton { pos, pressed, .. } => {
                    // A press goes where it is; a release, to where the press went.
                    let here = shared.under(ctx, *pos);
                    let to = if *pressed { here } else { shared.capture.unwrap_or(here) };
                    if *pressed {
                        if shared.capture.is_none() && shared.keyboard != to {
                            surrender(&shared.ctx(ctx, shared.keyboard));
                            shared.keyboard = to;
                        }
                        shared.capture = Some(to);
                    } else {
                        // shortcut: one button at a time; a chord released in another order ends the capture early.
                        shared.capture = None;
                    }
                    to
                }
                Event::MouseWheel { .. } => shared.hover,
                Event::PointerGone => {
                    shared.capture = None;
                    let hover = std::mem::take(&mut shared.hover);
                    shared.send(hover, Event::PointerGone, &mut window_events);
                    continue;
                }
                Event::WindowFocused(_) | Event::SystemTheme(_) => {
                    for slot in shared.slots.iter_mut().filter(|s| s.alive) {
                        slot.events.push(event.clone());
                    }
                    Owner::Window
                }
                Event::Key { key: Key::Tab, pressed, modifiers, .. } if (modifiers.is_none() || *modifiers == Modifiers::SHIFT) => {
                    if *pressed {
                        match shared.tab(ctx, modifiers.shift) {
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
                Event::AccessKitActionRequest(request) => {
                    let found = shared.slots.iter().enumerate().find_map(|(i, s)| s.nodes.get(&request.target_node).map(|n| (i, *n)));
                    match found {
                        Some((i, node)) => {
                            let mut request = request.clone();
                            request.target_node = node;
                            if request.action == accesskit::Action::Focus && shared.keyboard != Owner::Pane(i) {
                                surrender(&shared.ctx(ctx, shared.keyboard));
                                shared.keyboard = Owner::Pane(i);
                            }
                            shared.slots[i].events.push(Event::AccessKitActionRequest(request));
                            continue;
                        }
                        None => Owner::Window,
                    }
                }
                // shortcut: touches, pinches and drops stay with the window; a pane that scrolls by touch needs them.
                _ => Owner::Window,
            };
            let event = match (to, event) {
                (Owner::Window, event) => event,
                (Owner::Pane(i), Event::PointerMoved(pos)) => Event::PointerMoved(shared.slots[i].to_pane(ctx, pos)),
                (Owner::Pane(i), Event::PointerButton { pos, button, pressed, modifiers }) => {
                    Event::PointerButton { pos: shared.slots[i].to_pane(ctx, pos), button, pressed, modifiers }
                }
                (Owner::Pane(i), Event::MouseWheel { unit, delta, phase, source, modifiers }) => {
                    let window_ppp = ctx.native_pixels_per_point().unwrap_or(1.0) * ctx.zoom_factor();
                    let delta = if unit == egui::MouseWheelUnit::Point { delta * window_ppp / shared.slots[i].ppp } else { delta };
                    Event::MouseWheel { unit, delta, phase, source, modifiers }
                }
                (_, event) => event,
            };
            shared.send(to, event, &mut window_events);
        }
        input.events = window_events;
    }

    fn output_hook(&mut self, ctx: &Context, output: &mut FullOutput) {
        let mut shared = lock(&self.shared);
        shared.window_tab = ctx.memory(|m| tab_ends(m.focus_candidates()));
        let (hover, capture, keyboard) = (shared.hover, shared.capture, shared.keyboard);
        let mut drag = false;
        // Panes placed before the window's content come first among the root's children, in their order.
        let mut before = 0;
        for (i, slot) in shared.slots.iter_mut().enumerate() {
            let Some(pane) = slot.out.take() else { continue };
            output.shapes.extend(pane.shapes);
            let to_window = |r: Rect| Rect::from_min_max(pane.origin + r.min.to_vec2() * pane.scale, pane.origin + r.max.to_vec2() * pane.scale);
            let platform = &mut output.platform_output;
            if hover == Owner::Pane(i) || capture == Some(Owner::Pane(i)) {
                platform.cursor_icon = pane.platform.cursor_icon;
            }
            if keyboard == Owner::Pane(i)
                && let Some(mut ime) = pane.platform.ime
            {
                ime.rect = to_window(ime.rect);
                ime.cursor_rect = to_window(ime.cursor_rect);
                ime.owner = slot.salt.with(ime.owner);
                platform.ime = Some(ime);
            }
            platform.commands.extend(pane.platform.commands);
            if let Some(viewport) = pane.viewport
                && let Some(root) = output.viewport_output.get_mut(&ViewportId::ROOT)
            {
                // The system takes the pointer for a window drag and may never report the release.
                if viewport.commands.iter().any(|c| matches!(c, egui::ViewportCommand::StartDrag)) {
                    drag = true;
                    slot.events.push(Event::PointerGone);
                }
                root.commands.extend(viewport.commands);
                root.repaint_delay = root.repaint_delay.min(viewport.repaint_delay);
            }
            if let (Some(window_tree), Some(pane_tree)) = (output.platform_output.accesskit_update.as_mut(), pane.platform.accesskit_update) {
                let place = (pane.origin, pane.scale, pane.bounds);
                let at = (slot.place == Place::Before).then(|| {
                    before += 1;
                    before - 1
                });
                slot.nodes = merge_tree(window_tree, pane_tree, slot.salt, place, at, keyboard == Owner::Pane(i));
            }
        }
        if drag {
            shared.capture = None;
        }
    }
}

/// Puts `pane`'s nodes under the root of `window`, as its child number `at` or, with `None`, its last child, and
/// returns window node → pane node.
fn merge_tree(
    window: &mut accesskit::TreeUpdate,
    pane: accesskit::TreeUpdate,
    salt: Id,
    (origin, scale, b): (Pos2, f32, Rect),
    at: Option<usize>,
    focused: bool,
) -> HashMap<NodeId, NodeId> {
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
        let mut children: Vec<NodeId> = root.children().to_vec();
        children.insert(at.unwrap_or(children.len()).min(children.len()), map(pane_root));
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

    /// A window 800 × 600 pixels at `zoom`, with a "Title" pane at zoom 1 over its top 40 pixels, a "Status" pane
    /// at zoom 1 over its bottom 30 pixels, and a "Page" button of its own between them.
    struct Window {
        ctx: Context,
        title: Pane,
        status: Pane,
        clicked: Vec<&'static str>,
    }

    impl Window {
        fn new(zoom: f32) -> Self {
            let ctx = Context::default();
            ctx.enable_accesskit();
            ctx.set_zoom_factor(zoom);
            let title = Pane::new(&ctx, "title", Place::Before);
            let status = Pane::new(&ctx, "status", Place::After);
            Self { ctx, title, status, clicked: Vec::new() }
        }

        fn frame(&mut self, events: Vec<Event>) -> FullOutput {
            let raw = RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0) / self.ctx.zoom_factor())),
                events,
                ..Default::default()
            };
            let (title, status, clicked) = (&self.title, &self.status, &mut self.clicked);
            let mut out = self.ctx.run_ui(raw, |ui| {
                let ppp = ui.ctx().pixels_per_point();
                let top = ui.max_rect().min;
                title.show(ui.ctx(), 1.0, top, Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 40.0)), |ui| {
                    if ui.button("Title").clicked() {
                        clicked.push("Title");
                    }
                });
                let bottom = egui::pos2(top.x, 570.0 / ppp);
                status.show(ui.ctx(), 1.0, bottom, Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 30.0)), |ui| {
                    if ui.button("Status").clicked() {
                        clicked.push("Status");
                    }
                });
                // The title's 40 pixels, in window points.
                ui.add_space(40.0 / ppp + 20.0);
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

        /// The size in pixels of what the panes draw.
        fn pane_pixels(&mut self) -> Vec2 {
            let out = self.frame(Vec::new());
            let shared = lock(&self.title.shared);
            let mapped: Vec<TextureId> = shared.slots.iter().flat_map(|s| s.textures.values().copied()).collect();
            drop(shared);
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
    fn panes_keep_their_pixel_size_and_their_clicks_when_the_window_zooms() {
        let mut sizes = Vec::new();
        for zoom in [1.0, 2.0, 0.5] {
            let mut w = Window::new(zoom);
            for _ in 0..3 {
                w.frame(Vec::new());
            }
            sizes.push(w.pane_pixels());
            for label in ["Title", "Status"] {
                assert!(w.find(label).is_positive(), "the {label} pane's button is in the window's tree at zoom {zoom}");
            }
            for label in ["Title", "Page", "Status"] {
                w.click(label);
            }
            assert_eq!(w.clicked, ["Title", "Page", "Status"], "each click lands in its own context at zoom {zoom}");
        }
        assert!(sizes.iter().all(|s| (*s - sizes[0]).length() < 0.5 && s.y > 500.0), "the panes draw at the same pixel size: {sizes:?}");
    }

    #[test]
    fn tab_goes_round_the_title_the_page_and_the_status_bar() {
        let mut w = Window::new(2.0);
        for _ in 0..3 {
            w.frame(Vec::new());
        }
        let tab = |shift| Event::Key {
            key: Key::Tab,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: if shift { Modifiers::SHIFT } else { Modifiers::NONE },
        };
        let mut focused = Vec::new();
        for shift in [false, false, false, false, true] {
            w.frame(vec![tab(shift)]);
            let out = w.frame(Vec::new());
            let tree = out.platform_output.accesskit_update.expect("accesskit is on");
            let label = tree.nodes.iter().find(|(id, _)| *id == tree.focus).and_then(|(_, n)| n.label().map(str::to_owned));
            focused.push(label.unwrap_or_default());
        }
        assert_eq!(focused, ["Page", "Status", "Title", "Page", "Title"], "Tab goes round the three contexts, Shift+Tab back");
        // The tree reads in the same order.
        let out = w.frame(Vec::new());
        let tree = out.platform_output.accesskit_update.expect("accesskit is on");
        let root = tree.tree.map(|t| t.root);
        let order: Vec<accesskit::Role> = tree
            .nodes
            .iter()
            .find(|(id, _)| Some(*id) == root)
            .map(|(_, n)| n.children().iter().filter_map(|c| tree.nodes.iter().find(|(id, _)| id == c)).map(|(_, n)| n.role()).collect())
            .unwrap_or_default();
        assert_eq!(order.first(), Some(&accesskit::Role::Pane), "the title pane reads first: {order:?}");
        assert_eq!(order.last(), Some(&accesskit::Role::Pane), "the status pane reads last: {order:?}");
    }
}
