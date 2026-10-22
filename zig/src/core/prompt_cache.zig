//! Exact prompt reuse for any family and backend: states kept at prompt-pass chunk ends, found by their tokens.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");

/// A family's copy of one state.
pub const Saved = *anyopaque;

/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
        /// The storage a kept state holds: new storage up to `bytes`, or the spare storage it took (null: `bytes`).
        charged: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Free storage the family keeps for later saves, inside the budget (null: none).
        spare: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Free spare storage until at most `room` bytes stay spare.
        trim: ?*const fn (ptr: *anyopaque, room: u64) void = null,
        /// Whether a save at `at` would take spare storage (no new storage).
        reuses: ?*const fn (ptr: *anyopaque, at: u32) bool = null,
        /// Write `saved` (its `tokens`) as learned state `key`, files of its own under `dir` (spill: known ones too).
        write: ?*const fn (ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64, tokens: []const u32) anyerror!void = null,
        /// Learned state `key` of `at` tokens (`tokens`), read back from its files under `dir` into new storage.
        read: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32, tokens: []const u32) anyerror!Saved = null,
        /// Learned state `key`'s files under `dir` removed (the cap needs room, or they no longer read back).
        forget: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64) void = null,
        /// The disk bytes a learned state at `at` takes at most, for the cap (null: `bytes`).
        disk_bytes: ?*const fn (ptr: *anyopaque, at: u32) u64 = null,
        /// The states the index holds when the store opens on `dir`: the family drops any other's files, says a Known.
        known: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, keys: []const u64) u8 = null,
        /// `write` only queues: the keys of writes finished since (each its error or null), all of them when `wait`.
        settled: ?*const fn (ptr: *anyopaque, keys: []u64, failures: []?anyerror, wait: bool) usize = null,
    };

    /// What `known` found: the index stands, none of it reads back any more (a peer's files changed), or no disk tier.
    pub const Known = enum(u8) { keep = 0, clear = 1, off = 2 };
};

/// What a family's states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep nothing: below it a mark's extra prompt call costs more than a later turn's reuse saves.
    min_prompt: u32 = 4096,
    /// Rows a resumed pass must still run past its state (DeepSeek's sliding window): a state nearer the end is passed over.
    tail: u32 = 0,
};

pub const Entry = struct {
    tokens: []u32, // the state's tokens and `lookahead` more
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the length of the prompt that kept it: a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
};

pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0 };

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// The entry a backend restores itself (null: the pass starts at 0) and the pass's marks.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };

pub const Store = struct {
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers
    spill: bool = false, // every kept state to the imprint, hashed: an evicted or restarted conversation resumes there
    min_spill: u32 = 0, // spill: a conversation's state shorter than this stays in memory (shared cuts go at any length)
    pending: std.ArrayList(Pending) = .empty, // spill writes the family queued: listed in the index once they finished

    const SHARED_KEYS = 64;

    /// A queued spill write's index record, made when it is queued (the entry and its tokens may be gone when it ends).
    const Pending = struct { key: u64, at: u32, n: u32, check: u64, starts: []const u32, bytes: u64, known: bool = false };

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        return .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
    }

    pub fn deinit(s: *Store) void {
        s.settle(true); // the queued writes end before their states go
        for (s.pending.items) |p| s.gpa.free(p.starts);
        s.pending.deinit(s.gpa);
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.held = 0;
    }

    fn free(s: *Store, e: *Entry) void {
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    }

    fn remove(s: *Store, i: usize) void {
        const e = s.entries.orderedRemove(i);
        s.held -= e.bytes;
        s.free(e);
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.held -| s.spare();
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or std.mem.indexOfScalar(u32, starts, at) != null);
    }

    /// The longest kept state `prompt` resumes exactly: its tokens a prefix of the prompt, at least one row and `tail` left.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |e| {
            if (e.tokens.len > prompt.len or e.at >= prompt.len or @as(usize, e.at) + s.rules.tail > prompt.len or !s.usable(e.at, starts)) continue;
            if (best != null and e.at <= best.?.at) continue;
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            best = e;
        }
        return best;
    }

    /// Before a prompt pass: restore the longest state `prompt` resumes (from 0: none or a failed copy) and plan its marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque) !Plan {
        const l = try s.lookup(a, prompt, history_len, shared, starts);
        const e = l.entry orelse return .{ .marks = l.marks };
        const ok = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| true else |err| blk: {
            note("restoring {d} tokens failed ({s}); prefilling from the start", .{ e.at, @errorName(err) });
            break :blk false;
        };
        const from = if (ok) e.at else 0;
        s.resumed(e, prompt, ok);
        s.reserve(prompt, null, l.marks); // restored: the state it resumed may go too (a peer restores before it drops)
        return .{ .from = from, .marks = l.marks };
    }

    /// begin without the restore, for backends that restore inside their own prompt pass and then call `resumed`.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32) !Lookup {
        s.settle(false);
        const e = s.recall(prompt, starts, s.find(prompt, starts));
        if (e == null) s.counts.misses += 1;
        const marks_ = try s.fitting(a, try s.marks(a, prompt, if (e) |x| x.at else 0, history_len, shared, starts, if (e) |x| x.last else &.{}));
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const at = if (s.rules.planned) floorStart(starts, w) else w;
            if (std.mem.indexOfScalar(u32, marks_, at) != null) s.noteShared(prompt[0 .. at + s.rules.lookahead]);
        }
        s.reserve(prompt, e, marks_);
        return .{ .entry = e, .marks = marks_ };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, prompt: []const u32, from: ?*Entry, marks_: []const u32) void {
        var need: u64 = 0;
        for (marks_) |at| need += s.family.vtable.bytes(s.family.ptr, at);
        while (s.held + need > s.budget) {
            const i = s.victimBut(from, prompt) orelse return;
            s.remove(i);
            s.counts.evicted += 1;
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        for (s.entries.items, 0..) |e, i| if (e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) {
            s.counts.failed += 1;
            return s.remove(i);
        };
    }

    fn sharedKey(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }

    fn noteShared(s: *Store, tokens: []const u32) void {
        const k = sharedKey(tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.gpa, k) catch {};
    }

    /// The marks whose state the budget can hold (a new slice in `a`); the rest are refused now, so no pass, nor a peer, copies them.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        for (marks_) |at| {
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget) try out.append(a, at) else {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        const i = std.mem.indexOfScalar(*Entry, s.entries.items, e) orelse return;
        if (!ok) {
            s.counts.failed += 1;
            return s.remove(i);
        }
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) floorStart(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            if (k > 0) { // the history's mark always; the others only away from it, each other and the resume point
                if (at - from < s.rules.min_gap) continue;
                const near = for (out.items) |o| {
                    if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// The prompt pass stands at `at`: keep its state for `prompt`, evicting to fit; refused (counted) past the budget.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32) bool {
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        const shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(prompt[0..n])) != null;
        for (s.entries.items) |e| if (e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n])) {
            e.used = s.clock; // the same state again: no copy
            e.shared = e.shared or shared;
            return true;
        };
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (f(s.family.ptr, at)) break;
            const free_ = s.spare();
            if (s.held + free_ + bytes <= s.budget) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            s.remove(s.victimBut(null, prompt).?); // its storage may come back spare and take this state
            s.counts.evicted += 1;
        }
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| bytes);
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const tokens = s.gpa.dupe(u32, prompt[0..n]) catch {
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const last = s.gpa.dupe(u32, prompt) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .tokens = tokens, .at = at, .saved = saved, .bytes = charged, .born = @intCast(prompt.len), .used = s.clock, .last = last, .shared = shared };
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.held += charged;
        s.counts.kept += 1;
        if (shared or s.spill) s.learn(e, starts);
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held); // spare storage only inside what the budget leaves
        return true;
    }

    /// A shared cut's state written to disk once (--learn): later sessions read it back, after a restart too.
    fn learn(s: *Store, e: *const Entry, starts: []const u32) void {
        const im = s.imprint orelse return;
        const write = s.family.vtable.write orelse return;
        if (s.spill and !e.shared and e.at < s.min_spill) return;
        const queued = s.spill and s.family.vtable.settled != null;
        if (queued) s.settle(false);
        const key = imprint.Imprint.keyOf(e.tokens);
        const bytes = if (s.family.vtable.disk_bytes) |f| f(s.family.ptr, e.at) else e.bytes;
        const known = im.has(key);
        if ((known and !s.spill) or bytes > im.cap) return;
        if (queued and s.pendingOf(key) != null) return; // its write is under way
        if (!known) while (!im.fits(bytes + s.pendingBytes())) s.unlearn(im, im.victim() orelse return); // least recently used first
        write(s.family.ptr, e.saved, im.dir, key, e.tokens) catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        if (known) {
            if (queued) s.watch(key, e); // a rewrite that fails is said when it ends, as the synchronous one is said at once
            return im.touch(key); // already on disk: the family noted it for this kept state
        }
        if (queued) return s.queue(key, e, starts, bytes);
        const added = if (s.spill) im.addHashed(key, e.at, e.tokens, starts, bytes) else im.add(key, e.at, e.tokens, starts, bytes);
        added catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        if (!@import("builtin").is_test) std.log.info("prompt cache: learned {d} tokens to disk", .{e.at});
    }

    /// A queued write's record kept until the family says it finished (no room for it: it is never listed).
    fn queue(s: *Store, key: u64, e: *const Entry, starts: []const u32, bytes: u64) void {
        const st = s.gpa.dupe(u32, starts) catch return note("learning {d} tokens failed ({s})", .{ e.at, "OutOfMemory" });
        const p: Pending = .{ .key = key, .at = e.at, .n = @intCast(e.tokens.len), .check = imprint.Imprint.checkOf(e.tokens), .starts = st, .bytes = bytes };
        s.pending.append(s.gpa, p) catch {
            s.gpa.free(st);
            note("learning {d} tokens failed ({s})", .{ e.at, "OutOfMemory" });
        };
    }

    /// A known state's queued rewrite: the index lists it already, so only its failure matters when it ends.
    fn watch(s: *Store, key: u64, e: *const Entry) void {
        s.pending.append(s.gpa, .{ .key = key, .at = e.at, .n = 0, .check = 0, .starts = &.{}, .bytes = 0, .known = true }) catch {}; // (no room: a failure goes unsaid)
    }

    fn pendingOf(s: *const Store, key: u64) ?usize {
        for (s.pending.items, 0..) |p, i| if (p.key == key) return i;
        return null;
    }

    /// The disk the queued writes will take: the cap counts them before they are listed.
    fn pendingBytes(s: *const Store) u64 {
        var n: u64 = 0;
        for (s.pending.items) |p| n += p.bytes;
        return n;
    }

    /// The family's queued writes that finished, each listed in the index (or noted, failed); `wait`: all of them.
    pub fn settle(s: *Store, wait: bool) void {
        const f = s.family.vtable.settled orelse return;
        var keys: [16]u64 = undefined;
        var failures: [16]?anyerror = undefined;
        while (true) {
            const n = f(s.family.ptr, &keys, &failures, wait);
            for (keys[0..n], failures[0..n]) |k, x| s.listed(k, x);
            if (n < keys.len) return;
        }
    }

    fn listed(s: *Store, key: u64, failure: ?anyerror) void {
        const p = s.pending.orderedRemove(s.pendingOf(key) orelse return); // (none: its record did not fit)
        defer s.gpa.free(p.starts);
        if (failure) |err| return note("learning {d} tokens failed ({s})", .{ p.at, @errorName(err) });
        if (p.known) return; // the index lists it already
        const im = s.imprint orelse return;
        im.addSummed(p.key, p.at, p.n, p.check, p.starts, p.bytes) catch |err| return note("learning {d} tokens failed ({s})", .{ p.at, @errorName(err) });
        if (!@import("builtin").is_test) std.log.info("prompt cache: learned {d} tokens to disk", .{p.at});
    }

    /// The imprint just opened: kept as its files are, cleared (a peer's files changed), or dropped (false: free it).
    pub fn adopt(s: *Store) !bool {
        const im = s.imprint orelse return false;
        const known = s.family.vtable.known orelse return true;
        const keys = try s.gpa.alloc(u64, im.metas.items.len);
        defer s.gpa.free(keys);
        for (im.metas.items, keys) |x, *k| k.* = x.key;
        const answer = known(s.family.ptr, im.dir, keys);
        if (answer == @intFromEnum(Snapshots.Known.keep)) return true;
        if (answer == @intFromEnum(Snapshots.Known.clear)) {
            for (keys) |k| s.unlearn(im, k);
            note("the learned states no longer read back; {d} forgotten", .{keys.len});
            return true;
        }
        s.imprint = null; // off
        return false;
    }

    /// Learned state `key` forgotten: its files (the family's), then its index record.
    fn unlearn(s: *Store, im: *imprint.Imprint, key: u64) void {
        if (s.family.vtable.forget) |f| f(s.family.ptr, im.dir, key);
        im.remove(key) catch |err| note("forgetting a learned state failed ({s})", .{@errorName(err)});
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, starts: []const u32, have: ?*Entry) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const upto = if (s.rules.tail == 0) prompt else prompt[0..(prompt.len + 1) -| s.rules.tail]; // `tail` left past it
        const m = im.best(upto, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        const read_ = m.read(); // a hashed state's tokens are the prompt's (best matched them)
        const tokens = if (m.hashed()) prompt[0..read_] else m.tokens;
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        while (bytes <= s.budget and s.held + s.spare() + bytes > s.budget) {
            s.remove(s.victimBut(have, prompt) orelse return have);
            s.counts.evicted += 1;
        }
        if (bytes > s.budget) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = &.{}, .shared = !m.hashed() }; // a spilled turn may be superseded
        e.tokens = s.gpa.dupe(u32, tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at, e.tokens) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            s.unlearn(im, key);
            return s.drop(e, have, false);
        };
        im.touch(key);
        s.entries.append(s.gpa, e) catch return s.drop(e, have, true);
        s.held += bytes;
        if (!@import("builtin").is_test) std.log.info("prompt cache: read {d} learned tokens from disk", .{m.at});
        return e;
    }

    /// A half-made recalled entry undone (`saved` once read); the lookup goes on with `have`.
    fn drop(s: *Store, e: *Entry, have: ?*Entry, saved: bool) ?*Entry {
        if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
        return have;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip` (a pass's resume): one a later prompt extends (never a shared cut), oldest first; else the oldest.
    fn victimBut(s: *const Store, skip: ?*Entry, prompt: []const u32) ?usize {
        var best: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            if (e.shared or e == skip) continue; // a shared cut is evicted only as the least recently used
            const now = prompt.len > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens); // this prompt extends it
            const moved_on = now or for (s.entries.items) |o| {
                if (o != e and o.born > e.born and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) break true;
            } else false;
            if (moved_on and (best == null or e.used < s.entries.items[best.?].used)) best = i;
        }
        if (best) |i| return i;
        var oldest: ?usize = null;
        for (s.entries.items, 0..) |e, i| if (e != skip and (oldest == null or e.used < s.entries.items[oldest.?].used)) {
            oldest = i;
        };
        return oldest;
    }
};

/// Under test: the lines `note` has said (how many, and the last in full), as the log would show them.
pub const Said = struct {
    count: usize = 0,
    buf: [200]u8 = undefined,
    len: usize = 0,

    pub fn last(x: *const Said) []const u8 {
        return x.buf[0..x.len];
    }
};
pub var said: Said = .{};

/// A line on the engine's log; tests keep it in `said` instead (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) {
        said.count += 1;
        said.len = if (std.fmt.bufPrint(&said.buf, "prompt cache: " ++ fmt, args)) |out| out.len else |_| said.buf.len;
        return;
    }
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// The chunk start at or before `at` (0 when none).
fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

test {
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_spill_test.zig");
}
