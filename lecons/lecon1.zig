const std = @import("std");

fn lireU64LE(octets: []const u8) u64 {
    var res: u64 = 0;
    var mul: u64 = 1;
    for (octets, 0..) |o, i| {
        std.debug.print("o {d}\n", .{o});
        res = res + o * mul;
        std.debug.print("res {d}\n", .{res});
        if (i + 1 < octets.len)
            mul = mul * 256;
    }
    return res;
}

fn lireU64LE_pow(octets: []const u8) u64 {
    var res: u64 = 0;
    var mul: u64 = 1;
    for (octets, 0..) |o, i| {
        mul = @as(u64, 1) << @intCast(8 * i);
        res = res + o * mul;
    }
    return res;
}

pub fn main() void {
    const entete = [_]u8{ 0xf8, 0xb4, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00 };
    const res = lireU64LE(&entete);
    const res2 = lireU64LE_pow(&entete);

    std.debug.print("res {d}\n", .{res});

    std.debug.print("res_pow {d}\n", .{res2});
}
