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

// E4M3, OCP "FN" variant: 1 sign, 4 exponent (bias 7), 3 mantissa.
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

// element[0] is the LOW nibble. Not deducible from the bytes: settled by
// correlating against the unquantised model, 0.9954 vs 0.0384 (outils/oracle.py).
pub fn unpack(byte: u8) [2]u4 {
    const low: u4 = @intCast(byte & 15);
    const high: u4 = @intCast(byte >> 4);
    return .{ low, high };
}

// One block: 16 E2M1 elements in 8 bytes, one E4M3 scale, one FP32 tensor scale.
// Both scales folded once per block: half the multiplications, one extra ULP.
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

// Same, with the 16 values read from a compile-time table. The win is the branch
// that disappears, which lets LLVM vectorise the loop: 0.76 -> 12.9 GB/s.
pub fn decodeBlockTable(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    for (bytes, 0..) |p, i| {
        const d_p = unpack(p);

        out[2 * i] = total_scale * E2M1_VALUES[d_p[0]];
        out[2 * i + 1] = total_scale * E2M1_VALUES[d_p[1]];
    }
}
