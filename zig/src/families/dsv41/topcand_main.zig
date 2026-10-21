//! tf-dsv41-topcand [ROWS [SEED]]: the device's candidate lists against sampling.topCandidates on random logits rows.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const sampling = dsv41.sampling;
const Ops = dsv41.ops.Ops;

const vocab = 129280; // the served vocabulary
const batch = 16; // a round's rows at most (round.default_rows)
const pad = 17; // the most columns a row's stride runs past its n
const tiles = (vocab + Ops.candidate_tile - 1) / Ops.candidate_tile;

const Style = enum { uniform, ties, zeros, mixed, boundary, extremes, nan };
const style_count = 7;

/// The NaNs a row can hold: quiet and signaling, both signs, a few payloads.
const nans = [_]u32{ 0x7fc00000, 0xffc00000, 0x7f800001, 0xffffffff, 0x7fffffff, 0xff800001 };

/// Vocabularies around the tile width and the candidate counts.
const sizes = [_]usize{ 64, 65, 100, 4095, 4096, 4097, 8192, 8193, 12289, 30000, 129279 };

/// A column of a row of n: often a tile's edge or the row's first or last.
fn column(rnd: std.Random, n: usize) usize {
    const edges = [_]usize{ 0, 1, 4095, 4096, 4097, 8191, 8192, n - 1, n - 2 };
    const edge = edges[rnd.uintLessThan(usize, edges.len)];
    return if (rnd.uintLessThan(u32, 3) == 0 and edge < n) edge else rnd.uintLessThan(usize, n);
}

/// The row's values for `style`; k is the candidate count the row is selected with.
fn fillRow(rnd: std.Random, style: Style, k: usize, row: []f32) void {
    const inf = std.math.inf(f32);
    const n = row.len;
    switch (style) {
        .uniform, .nan => for (row) |*v| {
            v.* = (rnd.float(f32) - 0.5) * 20;
        },
        .ties => for (row) |*v| {
            v.* = @floatFromInt(@as(i32, @intCast(rnd.uintLessThan(u32, 7))) - 3);
        },
        .zeros => for (row) |*v| {
            v.* = if (rnd.boolean()) -0.0 else 0.0;
        },
        .mixed => {
            for (row) |*v| {
                const r = rnd.uintLessThan(u32, 1000);
                v.* = if (r < 20) -inf else if (r < 25) inf else if (r < 60) -0.0 else if (r < 100) 0.0 else (rnd.float(f32) - 0.5) * 20;
            }
            // one value repeated at many columns, as the row's maximum
            for (0..1 + rnd.uintLessThan(u32, 90)) |_| row[column(rnd, n)] = 11.0;
        },
        .boundary => {
            for (row) |*v| v.* = (rnd.float(f32) - 0.5) * 18 - 1;
            // k - 3 to k + 3 copies of the maximum: the tie group straddles the last candidate
            for (0..@max(1, k + rnd.uintLessThan(u32, 7) -| 3)) |_| row[column(rnd, n)] = 9.0;
        },
        .extremes => {
            const table = [_]f32{ @bitCast(@as(u32, 0x7f7fffff)), @bitCast(@as(u32, 0xff7fffff)), @bitCast(@as(u32, 1)), @bitCast(@as(u32, 0x80000001)), @bitCast(@as(u32, 0x00800000)), @bitCast(@as(u32, 0x807fffff)), 0.0, -0.0, inf, -inf, 1.0, -1.0 };
            for (row) |*v| v.* = table[rnd.uintLessThan(usize, table.len)];
        },
    }
    if (style == .nan) {
        for (0..1 + rnd.uintLessThan(u32, 3)) |_| row[column(rnd, n)] = @bitCast(nans[rnd.uintLessThan(usize, nans.len)]);
        if (rnd.boolean()) row[column(rnd, n)] = if (rnd.boolean()) inf else -inf;
    }
}

fn hasNan(row: []const f32) bool {
    for (row) |v| if (v != v) return true;
    return false;
}

const Tally = struct {
    rows: usize = 0,
    nan_rows: usize = 0,
    differ: usize = 0, // rows without a NaN whose list is not the host's
    unflagged: usize = 0, // rows with a NaN the device did not flag
    flagged: usize = 0, // rows without one that it flagged
    styles: [style_count]usize = @splat(0),
    first: [320]u8 = undefined,
    first_len: usize = 0,

    fn note(t: *Tally, comptime fmt: []const u8, args: anytype) void {
        if (t.first_len == 0) t.first_len = (std.fmt.bufPrint(&t.first, fmt, args) catch return).len;
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const total = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 20000;
    const seed = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 20261021;
    if (!cuda.kernels.available) return error.NoKernelImages;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var ops = try Ops.load(&driver, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
    defer ops.unload();
    const stride = vocab + pad;
    var logits = try cuda.DeviceBuffer.alloc(&driver, batch * stride * 4);
    defer logits.free();
    var scratch = try cuda.DeviceBuffer.alloc(&driver, batch * tiles * sampling.max_candidates * @sizeOf(sampling.Candidate));
    defer scratch.free();
    var out = try cuda.DeviceBuffer.alloc(&driver, batch * sampling.max_candidates * @sizeOf(sampling.Candidate));
    defer out.free();
    const host = try a.alloc(f32, batch * stride);
    const got = try a.alloc(sampling.Candidate, batch * sampling.max_candidates);
    var want_v: [sampling.max_candidates]f32 = undefined;
    var want_i: [sampling.max_candidates]i64 = undefined;
    var prng: std.Random.DefaultPrng = .init(seed);
    const rnd = prng.random();
    var tally: Tally = .{};
    const started = std.Io.Timestamp.now(io, .awake);
    while (tally.rows < total) {
        const rows = @min(total - tally.rows, 1 + rnd.uintLessThan(usize, batch));
        const n: usize = switch (rnd.uintLessThan(u32, 10)) {
            0...5 => vocab,
            6 => sizes[rnd.uintLessThan(usize, sizes.len)],
            else => 64 + rnd.uintLessThan(usize, vocab - 63),
        };
        const k = @min(n, 21 + rnd.uintLessThan(usize, 44)); // 21 to 64
        const ld = n + if (rnd.boolean()) 0 else rnd.uintLessThan(usize, pad + 1);
        var styles: [batch]Style = undefined;
        for (0..rows) |r| {
            styles[r] = if (rnd.uintLessThan(u32, 8) == 0) .nan else @enumFromInt(rnd.uintLessThan(u8, style_count));
            fillRow(rnd, styles[r], k, host[r * ld ..][0..n]);
            for (host[r * ld + n ..][0 .. ld - n]) |*v| v.* = @bitCast(nans[0]); // (a read past n would flag the row)
        }
        try logits.upload(0, std.mem.sliceAsBytes(host[0 .. rows * ld]));
        try ops.candidateRows(stream, logits.ptr, ld, n, rows, k, scratch.ptr, out.ptr);
        try stream.synchronize();
        try out.download(0, std.mem.sliceAsBytes(got[0 .. rows * k]));
        for (0..rows) |r| {
            const row = host[r * ld ..][0..n];
            const list = got[r * k ..][0..k];
            tally.rows += 1;
            tally.styles[@intFromEnum(styles[r])] += 1;
            const flagged = for (list) |c| {
                if (c.id == sampling.nan_row) break true;
            } else false;
            if (hasNan(row)) {
                tally.nan_rows += 1;
                var all = true;
                for (list) |c| all = all and c.id == sampling.nan_row;
                if (!all) {
                    tally.unflagged += 1;
                    tally.note("row {d} ({s}, n {d}, k {d}) holds a NaN and is not flagged in every entry", .{ tally.rows - 1, @tagName(styles[r]), n, k });
                }
                continue;
            }
            if (flagged) {
                tally.flagged += 1;
                tally.note("row {d} ({s}, n {d}, k {d}) holds no NaN and is flagged", .{ tally.rows - 1, @tagName(styles[r]), n, k });
                continue;
            }
            if (sampling.topCandidates(row, k, &want_v, &want_i) != k) return error.HostListShort;
            for (list, want_v[0..k], want_i[0..k], 0..) |c, v, id, i| {
                if (c.id == id and @as(u32, @bitCast(c.value)) == @as(u32, @bitCast(v))) continue;
                tally.differ += 1;
                tally.note("row {d} ({s}, n {d}, k {d}) entry {d}: device ({x:0>8}, {d}), host ({x:0>8}, {d})", .{ tally.rows - 1, @tagName(styles[r]), n, k, i, @as(u32, @bitCast(c.value)), c.id, @as(u32, @bitCast(v)), id });
                break;
            }
        }
    }
    const seconds = @as(f64, @floatFromInt(started.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
    var out_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout.interface;
    try w.print("{{\"rows\": {d}, \"nan_rows\": {d}, \"differ\": {d}, \"nan_unflagged\": {d}, \"finite_flagged\": {d}, \"seconds\": {d:.1}, \"styles\": {{", .{ tally.rows, tally.nan_rows, tally.differ, tally.unflagged, tally.flagged, seconds });
    for (0..style_count) |i| try w.print("{s}\"{s}\": {d}", .{ if (i == 0) "" else ", ", @tagName(@as(Style, @enumFromInt(i))), tally.styles[i] });
    try w.print("}}, \"first\": \"{s}\"}}\n", .{tally.first[0..tally.first_len]});
    // what a served round's 16 rows cost each way: the host's scan of the logits, and the two launches
    for (host[0 .. batch * stride], 0..) |*v, i| v.* = @floatFromInt(i % 977);
    const t0 = std.Io.Timestamp.now(io, .awake);
    for (0..batch) |r| _ = sampling.topCandidates(host[r * stride ..][0..vocab], 28, &want_v, &want_i);
    const host_ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
    try logits.upload(0, std.mem.sliceAsBytes(host[0 .. batch * stride]));
    var e0 = try cuda.Event.init(&driver, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(&driver, true);
    defer e1.deinit();
    for (0..10) |_| try ops.candidateRows(stream, logits.ptr, stride, vocab, batch, 28, scratch.ptr, out.ptr);
    try e0.record(stream);
    for (0..100) |_| try ops.candidateRows(stream, logits.ptr, stride, vocab, batch, 28, scratch.ptr, out.ptr);
    try e1.record(stream);
    try e1.synchronize();
    try w.print("{{\"host_ms_16_rows\": {d:.3}, \"device_ms_16_rows\": {d:.3}}}\n", .{ host_ms, try cuda.Event.elapsedMs(e0, e1) / 100 });
    try w.flush();
    return if (tally.rows == total and tally.differ == 0 and tally.unflagged == 0 and tally.flagged == 0) 0 else 1;
}
