// Throughput of the three block decoders, on a working set that stays in L2.
const std = @import("std");
const nvfp4 = @import("nvfp4.zig");

const BLOCKS: usize = 4096; // 32 KB packed -> 256 KB of f32
const ROUNDS: usize = 2000;
const RUNS: usize = 7;

fn pass(comptime decode: fn (f32, u8, [8]u8, *[16]f32) void, packed_bytes: []const u8, scales: []const u8, out: []f32) void {
    for (0..scales.len) |i| {
        decode(1.0, scales[i], packed_bytes[8 * i ..][0..8].*, out[16 * i ..][0..16]);
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    const packed_bytes = try gpa.alloc(u8, 8 * BLOCKS);
    defer gpa.free(packed_bytes);
    const scales = try gpa.alloc(u8, BLOCKS);
    defer gpa.free(scales);
    const out = try gpa.alloc(f32, 16 * BLOCKS);
    defer gpa.free(out);

    var prng = std.Random.DefaultPrng.init(12345);
    const rnd = prng.random();
    rnd.bytes(packed_bytes);
    for (scales) |*s| s.* = rnd.intRangeAtMost(u8, 1, 126); // no NaN, no zero

    const names = [_][]const u8{ "table", "gather", "simd" };
    var best = [_]f64{ 1e9, 1e9, 1e9 };

    for (0..RUNS) |_| {
        inline for (.{ nvfp4.decodeBlockTable, nvfp4.decodeBlockGather, nvfp4.decodeBlockSimd }, 0..) |decode, k| {
            const t0 = std.Io.Clock.awake.now(io);
            for (0..ROUNDS) |_| pass(decode, packed_bytes, scales, out);
            const t1 = std.Io.Clock.awake.now(io);
            const s = @as(f64, @floatFromInt(std.Io.Timestamp.durationTo(t0, t1).toMicroseconds())) / 1e6;
            if (s < best[k]) best[k] = s;
            std.mem.doNotOptimizeAway(out[rnd.intRangeLessThan(usize, 0, out.len)]);
        }
    }

    const bytes: f64 = @floatFromInt(16 * BLOCKS * 4 * ROUNDS);
    std.debug.print("best of {d}, {d} KB working set\n", .{ RUNS, 16 * BLOCKS * 4 / 1024 });
    for (names, best) |name, s| {
        std.debug.print("  {s:<8} {d:.4} s   {d:.2} GB/s of output\n", .{ name, s, bytes / 1e9 / s });
    }
}
