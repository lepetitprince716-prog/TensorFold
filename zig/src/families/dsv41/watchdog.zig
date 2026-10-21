//! A rank's watchdog: a busy primitive that makes no progress for the limit ends its pair (lanes.zig pairBroken).
const std = @import("std");

/// Seconds a primitive may run without a tick when TF_DS_WATCHDOG_S is unset.
pub const default_limit_s: u64 = 120;

/// A primitive that stopped ticking: its name and the seconds since its last progress.
pub const Stall = struct { name: []const u8, seconds: u64 };

/// What the watchdog thread calls when a primitive stalls.
pub const Fire = *const fn (ctx: *anyopaque, stall: Stall) void;

/// Where the watchdog reads time: nanoseconds on a monotonic clock (a test sets its own by hand).
pub const Clock = struct {
    ctx: *anyopaque,
    now: *const fn (ctx: *anyopaque) u64,
};

/// TF_DS_WATCHDOG_S as the limit in seconds: unset or empty is the default, 0 turns the watchdog off.
pub fn limitFromText(text: ?[]const u8) error{BadWatchdog}!u64 {
    const t = text orelse return default_limit_s;
    if (t.len == 0) return default_limit_s;
    return std.fmt.parseInt(u64, t, 10) catch error.BadWatchdog;
}

pub fn limitFromEnv() error{BadWatchdog}!u64 {
    const v = std.c.getenv("TF_DS_WATCHDOG_S") orelse return limitFromText(null);
    return limitFromText(std.mem.span(v));
}

pub const Watchdog = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    clock: Clock,
    limit_ns: u64,
    fire_ctx: *anyopaque,
    fire: Fire,
    /// Guards active, reported, name and since: the primitive's thread writes them, the watchdog's thread reads them.
    lock: std.atomic.Value(bool) = .init(false),
    active: bool = false,
    reported: bool = false,
    name: []const u8 = "",
    since: u64 = 0,
    stopped: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    sleep_ms: u64 = 100,
    every: u32 = 10,

    /// Starts the watchdog thread: `fire` runs there once a primitive has made no progress for `limit_s` seconds.
    pub fn start(gpa: std.mem.Allocator, io: std.Io, limit_s: u64, fire_ctx: *anyopaque, fire: Fire) !*Watchdog {
        return startPaced(gpa, io, limit_s *| std.time.ns_per_s, 100, 10, fire_ctx, fire);
    }

    /// The thread sleeps `sleep_ms` at a time (so that stop returns at once) and checks every `every` sleeps.
    fn startPaced(gpa: std.mem.Allocator, io: std.Io, limit_ns: u64, sleep_ms: u64, every: u32, fire_ctx: *anyopaque, fire: Fire) !*Watchdog {
        const w = try gpa.create(Watchdog);
        errdefer gpa.destroy(w);
        w.* = .{ .gpa = gpa, .io = io, .clock = .{ .ctx = w, .now = awake }, .limit_ns = limit_ns, .fire_ctx = fire_ctx, .fire = fire, .sleep_ms = sleep_ms, .every = every };
        w.thread = try std.Thread.spawn(.{}, loop, .{w});
        return w;
    }

    /// Stops the thread and frees the watchdog. Call before the model whose rings it may abort goes away.
    pub fn stop(w: *Watchdog) void {
        w.stopped.store(true, .release);
        if (w.thread) |t| t.join();
        w.gpa.destroy(w);
    }

    /// A primitive starts: its window opens now.
    pub fn busy(w: *Watchdog, name: []const u8) void {
        const at = w.clock.now(w.clock.ctx);
        w.acquire();
        defer w.release();
        w.active = true;
        w.reported = false;
        w.name = name;
        w.since = at;
    }

    /// The primitive made progress (a prompt chunk, a round): its window restarts. Idle: nothing to restart.
    pub fn tick(w: *Watchdog) void {
        const at = w.clock.now(w.clock.ctx);
        w.acquire();
        defer w.release();
        if (!w.active) return;
        w.reported = false;
        w.since = at;
    }

    /// The primitive ended, or the rank waits for its next frame: nothing can stall.
    pub fn idle(w: *Watchdog) void {
        w.acquire();
        defer w.release();
        w.active = false;
    }

    /// The busy primitive that has not ticked for the limit, reported once until it ticks again; null: none.
    pub fn check(w: *Watchdog) ?Stall {
        if (w.limit_ns == 0) return null;
        const at = w.clock.now(w.clock.ctx);
        w.acquire();
        defer w.release();
        if (!w.active or w.reported) return null;
        const quiet = at -| w.since;
        if (quiet < w.limit_ns) return null;
        w.reported = true;
        return .{ .name = w.name, .seconds = quiet / std.time.ns_per_s };
    }

    fn acquire(w: *Watchdog) void {
        while (w.lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn release(w: *Watchdog) void {
        w.lock.store(false, .release);
    }

    fn awake(ctx: *anyopaque) u64 {
        const w: *Watchdog = @ptrCast(@alignCast(ctx));
        return @intCast(std.Io.Clock.awake.now(w.io).toNanoseconds());
    }

    fn loop(w: *Watchdog) void {
        var slices: u32 = 0;
        while (!w.stopped.load(.acquire)) {
            std.Io.sleep(w.io, .fromMilliseconds(@intCast(w.sleep_ms)), .awake) catch {};
            slices += 1;
            if (slices < w.every) continue;
            slices = 0; // a check every second by default, the sleeps short so that stop returns at once
            if (w.stopped.load(.acquire)) break;
            if (w.check()) |stall| w.fire(w.fire_ctx, stall);
        }
    }
};

const testing = std.testing;
const sec = std.time.ns_per_s;

/// A clock the test sets by hand.
const Fake = struct {
    at: u64 = 0,

    fn now(ctx: *anyopaque) u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return f.at;
    }

    fn clock(f: *Fake) Clock {
        return .{ .ctx = f, .now = now };
    }
};

fn never(_: *anyopaque, _: Stall) void {}

/// A watchdog on the fake clock, no thread.
fn watcher(f: *Fake, limit_s: u64) Watchdog {
    return .{ .gpa = testing.allocator, .io = testing.io, .clock = f.clock(), .limit_ns = limit_s * sec, .fire_ctx = f, .fire = never };
}

test "a primitive with no tick fires at the limit and not a second before, once" {
    var f: Fake = .{ .at = 1000 * sec };
    var w = watcher(&f, 120);
    w.busy("verify");
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    const stall = w.check().?;
    try testing.expectEqualStrings("verify", stall.name);
    try testing.expectEqual(@as(u64, 120), stall.seconds);
    f.at += 600 * sec;
    try testing.expect(w.check() == null); // said once
    w.tick(); // progress after all: the next stall is reported again
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqual(@as(u64, 120), w.check().?.seconds);
}

test "ticks keep a primitive alive however long it runs, and the limit counts from the last one" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    w.busy("fill");
    for (0..1000) |_| { // a prompt pass of a day and more, a chunk every 100 s
        f.at += 100 * sec;
        try testing.expect(w.check() == null);
        w.tick();
    }
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqualStrings("fill", w.check().?.name);
}

test "idle never fires, a tick while idle arms nothing, and the next busy opens a fresh window" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    try testing.expect(w.check() == null); // never busy
    w.busy("verify");
    w.idle(); // the primitive ended
    f.at += 100000 * sec;
    try testing.expect(w.check() == null);
    w.tick();
    f.at += 100000 * sec;
    try testing.expect(w.check() == null);
    w.busy("pass");
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqualStrings("pass", w.check().?.name);
}

test "a primitive that ends in time leaves nothing for the next one to inherit" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    w.busy("verify");
    f.at += 100 * sec;
    w.idle();
    f.at += 50 * sec; // 150 s since the first busy, 50 s since it ended
    w.busy("pass");
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqualStrings("pass", w.check().?.name);
}

test "a limit of 0 is off" {
    var f: Fake = .{};
    var w = watcher(&f, 0);
    w.busy("verify");
    f.at += 100000 * sec;
    try testing.expect(w.check() == null);
}

test "the limit's text: unset or empty is 120 s, 0 is off, anything else but whole seconds is refused" {
    try testing.expectEqual(@as(u64, 120), try limitFromText(null));
    try testing.expectEqual(@as(u64, 120), try limitFromText(""));
    try testing.expectEqual(@as(u64, 0), try limitFromText("0"));
    try testing.expectEqual(@as(u64, 45), try limitFromText("45"));
    for ([_][]const u8{ "x", "-3", "1.5", "12s", " 5" }) |bad| try testing.expectError(error.BadWatchdog, limitFromText(bad));
}

test "the thread starts and stops with nothing stalled" {
    var ctx: u8 = 0;
    const w = try Watchdog.start(testing.allocator, testing.io, 3600, &ctx, never);
    w.busy("verify");
    w.tick();
    w.idle();
    Watchdog.stop(w);
}

/// What the thread's fire calls leave behind, for the test that runs it on the real clock.
const Fires = struct {
    n: std.atomic.Value(u32) = .init(0),
    last: []const u8 = "",

    fn fire(ctx: *anyopaque, stall: Stall) void {
        const f: *Fires = @ptrCast(@alignCast(ctx));
        f.last = stall.name;
        _ = f.n.fetchAdd(1, .release);
    }

    fn wait(f: *const Fires, want: u32) !void {
        for (0..1000) |_| { // up to 5 s
            if (f.n.load(.acquire) >= want) return;
            std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
        }
        return error.Timeout;
    }

    fn settle(f: *const Fires, want: u32) !void {
        std.Io.sleep(testing.io, .fromMilliseconds(120), .awake) catch {};
        try testing.expectEqual(want, f.n.load(.acquire));
    }
};

test "the thread fires once for a stalled primitive, again after a tick, and not while idle (the real clock)" {
    var fires: Fires = .{};
    const w = try Watchdog.startPaced(testing.allocator, testing.io, 40 * std.time.ns_per_ms, 5, 2, &fires, Fires.fire);
    w.busy("verify");
    try fires.wait(1);
    try testing.expectEqualStrings("verify", fires.last);
    try fires.settle(1); // the stall is said once
    w.tick();
    try fires.wait(2);
    w.idle();
    try fires.settle(2);
    Watchdog.stop(w);
}
