const std = @import("std");

// E4M3 (variante FN de l'OCP) : 1 bit de signe, 4 bits d'exposant (biais 7),
// 3 bits de mantisse. Pas d'infini. NaN = exposant 15 et mantisse 7.
fn decodeE4M3(octet: u8) f32 {
    const sign = octet >> 7 != 0; //last bit
    const decimal = octet & 7; //first 3 bits
    const f_decimal: f32 = @floatFromInt(decimal);
    const ee: i8 = @intCast((octet >> 3) & 15);

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

pub fn main() void {
    // test exhaustif : les 256 codes
    const octets = [_]u8{ 0x00, 0x38, 0x7E, 0x01, 8 };

    std.debug.print("out = ", .{});
    for (octets) |o| {
        std.debug.print("{any}, ", .{decodeE4M3(o)});
    }

    for (0..256) |i| {
        std.debug.print("{any}, ", .{decodeE4M3(@intCast(i))});
    }
}
