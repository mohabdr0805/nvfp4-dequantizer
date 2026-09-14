const std = @import("std");
const nvp4 = @import("nvfp4.zig");
const safetensors = @import("safetensors.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    _ = io;
    _ = gpa;
}
