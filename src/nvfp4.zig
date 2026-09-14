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

// E4M3 (variante FN de l'OCP) : 1 bit de signe, 4 bits d'exposant (biais 7),
// 3 bits de mantisse. Pas d'infini. NaN = exposant 15 et mantisse 7.
pub fn decodeE4M3(octet: u8) f32 {
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

// CONVENTION DE NIBBLES : element[0] = nibble BAS, element[1] = nibble HAUT.
// Choix, pas deduction : les octets ne disent pas lequel porte l'indice pair.
// A valider contre une reference a l'etape 3d.
pub fn depaquete(octet: u8) [2]u4 {
    const faible: u4 = @intCast(octet & 15);
    const fort: u4 = @intCast(octet >> 4);
    return .{ faible, fort };
}

// Un bloc NVFP4 : 16 elements E2M1 dans 8 octets, 1 octet d'echelle E4M3,
// et l'echelle FP32 commune a tout le tenseur.
//     valeur = globale * decodeE4M3(echelle) * decodeE2M1(element)
pub fn decodeBloc(globale: f32, echelle: u8, paquet: [8]u8, sortie: *[16]f32) void {
    const echelle_decode = decodeE4M3(echelle);
    const echelle_totale = globale * echelle_decode;

    for (paquet, 0..) |p, i| {
        const d_p = depaquete(p);

        sortie[2 * i] = echelle_totale * decodeE2M1(d_p[0]);
        sortie[2 * i + 1] = echelle_totale * decodeE2M1(d_p[1]);
    }
}

const TABLE: [16]f32 = blk: {
    var t: [16]f32 = undefined;
    for (0..16) |i| t[i] = decodeE2M1(i);
    break :blk t;
};

//decode NVFP4 mais en prenans le tableau en comptime
pub fn decodeBloc_inline(globale: f32, echelle: u8, paquet: [8]u8, sortie: *[16]f32) void {
    const echelle_decode = decodeE4M3(echelle);
    const echelle_totale = globale * echelle_decode;

    for (paquet, 0..) |p, i| {
        const d_p = depaquete(p);

        sortie[2 * i] = echelle_totale * TABLE[d_p[0]];
        sortie[2 * i + 1] = echelle_totale * TABLE[d_p[1]];
    }
}
