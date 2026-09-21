const std = @import("std");

pub fn decodeE2M1(code: u4) f32 {
    const sign = code >> 3 != 0;
    const decimal = code & 1;
    const f_decimal: f32 = @floatFromInt(decimal);

    const ee = (code >> 1) & 3;

    var res: f32 = 0;

    if (ee == 0) {
        res = f_decimal * 0.5;
    } else {
        const shift: f32 = @floatFromInt(@as(u4, 1) << @intCast(ee - 1));
        res = (1 + f_decimal / 2) * shift;
    }

    res = if (sign) -res else res;

    return res;
}

// No infinity; NaN only when exponent is 15 and mantissa 7, hence max 448.
pub fn decodeE4M3(byte: u8) f32 {
    const sign = byte >> 7 != 0; // sign bit
    const decimal = byte & 7; // mantissa
    const f_decimal: f32 = @floatFromInt(decimal);
    const ee: i8 = @intCast((byte >> 3) & 15);

    var res: f32 = 0;

    if (ee == 0) {
        const shift: f32 = std.math.ldexp(@as(f32, 1.0), -6);
        res = (f_decimal / 8) * shift;
    } else if (ee == 15 and decimal == 7) {
        return std.math.nan(f32);
    } else {
        const shift: f32 = std.math.ldexp(@as(f32, 1.0), ee - 7);
        res = (1 + f_decimal / 8) * shift;
    }

    res = if (sign) -res else res;

    return res;
}

// element[0] is the LOW nibble.
pub fn unpack(byte: u8) [2]u4 {
    const low: u4 = @intCast(byte & 15);
    const high: u4 = @intCast(byte >> 4);
    return .{ low, high };
}

// One block: 16 E2M1 elements in 8 bytes, one E4M3 scale, one FP32 tensor scale.
pub fn decodeBlock(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    for (bytes, 0..) |p, i| {
        const d_p = unpack(p);

        out[2 * i] = total_scale * decodeE2M1(d_p[0]);
        out[2 * i + 1] = total_scale * decodeE2M1(d_p[1]);
    }
}

const E2M1_VALUES: [16]f32 = blk: {
    var t: [16]f32 = undefined;
    for (0..16) |i| t[i] = decodeE2M1(i);
    break :blk t;
};

pub fn decodeBlockTable(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    for (bytes, 0..) |p, i| {
        const d_p = unpack(p);

        out[2 * i] = total_scale * E2M1_VALUES[d_p[0]];
        out[2 * i + 1] = total_scale * E2M1_VALUES[d_p[1]];
    }
}

pub fn decodeBlockGather(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    const broadcast = @shuffle(u8, bytes, undefined, @Vector(16, u8){ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 });
    const mask: @Vector(16, bool) = .{ true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false };

    const low = broadcast & @as(@Vector(16, u8), @splat(15));
    const high = broadcast >> @as(@Vector(16, u8), @splat(4));

    const tmp: @Vector(16, u8) = @select(u8, mask, low, high);
    var res = @as(@Vector(16, f32), @splat(0));

    inline for (0..16) |i| {
        res[i] = E2M1_VALUES[tmp[i]];
    }

    res = res * @as(@Vector(16, f32), @splat(total_scale));

    out.* = res;
}

pub fn decodeBlockSimd(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    const broadcast = @shuffle(u8, bytes, undefined, @Vector(16, u8){ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 });
    const mask: @Vector(16, bool) = .{ true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false };

    const low = broadcast & @as(@Vector(16, u8), @splat(15));
    const high = broadcast >> @as(@Vector(16, u8), @splat(4));

    const tmp: @Vector(16, u32) = @select(u8, mask, low, high);

    var sign: @Vector(16, u32) = (tmp >> @as(@Vector(16, u8), @splat(3)));
    const decimal: @Vector(16, u32) = tmp & @as(@Vector(16, u8), @splat(1));

    const ee: @Vector(16, u32) = (tmp >> @as(@Vector(16, u8), @splat(1))) & @as(@Vector(16, u8), @splat(3));

    sign = sign << @as(@Vector(16, u8), @splat(31));
    const n_decimal = decimal << @as(@Vector(16, u8), @splat(22));

    const ee_zero_decimal = ee + (@as(@Vector(16, u32), @splat(63)) << @as(@Vector(16, u32), @splat(24)));
    const ee_zero = @select(u32, decimal == @as(@Vector(16, u32), @splat(0)), ee, ee_zero_decimal);

    const ee_nonzero = (ee + @as(@Vector(16, u8), @splat(126))) << @as(@Vector(16, u8), @splat(23));
    const ee_nonzero_decimal = ee_nonzero | n_decimal;

    const u_decoded: @Vector(16, u32) = @select(u32, ee == @as(@Vector(16, u32), @splat(0)), ee_zero, ee_nonzero_decimal);

    const decoded: @Vector(16, u32) = sign | u_decoded;
    var res: @Vector(16, f32) = @as(@Vector(16, f32), @bitCast(decoded));
    res = res * @as(@Vector(16, f32), @splat(total_scale));

    out.* = res;
}

pub const TENSOR_SCALES = [_]f32{ 1.0, 0.5, 0.0078125, 3.7e-3, 448.0 };

// A block filled with one byte covers all 16 codes in 16 turns, in both nibble
// positions. Bit patterns are compared, not floats: -0.0 and +0.0 must not pass.
fn matchesTable(comptime candidate: fn (f32, u8, [8]u8, *[16]f32) void) !void {
    var expected: [16]f32 = undefined;
    var got: [16]f32 = undefined;

    for (TENSOR_SCALES) |g| {
        for (0..256) |s| {
            for (0..256) |b| {
                const bytes: [8]u8 = @splat(@intCast(b));
                decodeBlockTable(g, @intCast(s), bytes, &expected);
                candidate(g, @intCast(s), bytes, &got);
                for (0..16) |i| {
                    const a: u32 = @bitCast(expected[i]);
                    const o: u32 = @bitCast(got[i]);
                    if (a != o) return error.Divergence;
                }
            }
        }
    }
}

test "decodeBlockGather matches the table" {
    try matchesTable(decodeBlockGather);
}

test "decodeBlockSimd matches the table" {
    try matchesTable(decodeBlockSimd);
}
