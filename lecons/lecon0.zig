const std = @import("std");

fn triple(x: i32) i32 {
    return x * 3;
}

fn somme(octets: []const u8) u32 {
    var res: u32 = 0;
    for (octets) |o| {
        res = res + o;
    }
    return res;
}

pub fn main() void {
    const x: i32 = 5;
    const res: i32 = triple(x);

    std.debug.print("x {d}\n", .{x});
    std.debug.print("res {d}\n", .{res});

    const list = [_]u8{ 10, 20, 30, 40 };
    const sum = somme(&list);

    std.debug.print("sum {d}\n", .{sum});
}
