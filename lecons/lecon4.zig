const std = @import("std");

fn decodeE2M1(code: u4) f32 {
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

pub fn main() void {
    const code: u4 = 10; //0b0101
    _ = code;

    std.debug.print("res : ", .{});
    for (0..16) |i| {
        const res = decodeE2M1(@intCast(i));
        std.debug.print("{d}, ", .{res});
    }
}
