const std = @import("std");
pub fn main() !void {
    const v: @Vector(4, f32) = .{ 1.0, 2.0, 3.0, 4.0 };
    const s = @reduce(.Add, v);
    std.debug.print("somme SIMD = {d}\n", .{s});
}
