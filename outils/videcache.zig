// Vide le cache de fichiers du systeme, en saturant la RAM puis en la rendant.
//
//     zig run -OReleaseFast outils/videcache.zig
//
// Necessaire avant toute mesure de lecture : sur une machine a 32 Go, un modele de
// 5 Go relu plusieurs fois donne 6 Go/s, soit trois fois le plafond du disque. Relire
// un autre gros fichier ne suffit pas a le chasser -- essaye le 19/09, sans effet.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const GO: usize = 1024 * 1024 * 1024;
    var liste: [26][]u8 = undefined;
    var n: usize = 0;
    while (n < liste.len) : (n += 1) {
        liste[n] = gpa.alloc(u8, GO) catch break;
        @memset(liste[n], 1);
    }
    std.debug.print("{d} Go touches\n", .{n});
    for (liste[0..n]) |b| gpa.free(b);
}
