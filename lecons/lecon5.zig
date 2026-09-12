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

// 0xBA, renvoie (B, A), convention a verifier
// Renvoie mtn (A, B) apres verification de la convention
fn depaquete(octet: u8) [2]u4 {
    const a: u4 = @intCast(octet & 15);
    const b: u4 = @intCast(octet >> 4);
    return .{ a, b };
}

pub fn main() void {
    const octets = [_]u8{ 0x10, 0x32, 0xB4 };
    var decodes: [octets.len * 2]f32 = undefined;

    std.debug.print("out : ", .{});
    for (octets, 0..) |o, i| {
        const out = depaquete(o);
        std.debug.print("{any}, ", .{out});
        decodes[2 * i] = decodeE2M1(out[0]);
        decodes[2 * i + 1] = decodeE2M1(out[1]);
    }
    std.debug.print("\ndecodes : {any}", .{decodes});
}
