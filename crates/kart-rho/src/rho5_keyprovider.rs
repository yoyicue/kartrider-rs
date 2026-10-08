//! Rho5KeyProvider — the `.rho5` keystream, a custom AES-like stream cipher ported
//! clean-room (line-by-line) from KartLibrary's disassembly-reverse-engineered
//! source. `InitFromKey` is the key schedule; `refresh` produces 16 `u32`
//! keystream words; `next_sub_num` hands them out. Validated against C# reference
//! values (see tests). Indexing keeps the original `offset>>2` form to match the C#.

use crate::rho5_tables::CONST_1;

pub struct Rho5KeyProvider {
    pub var_0: [u32; 0x14], // state (20 u32)
    pub var_1: [u32; 0x10], // keystream output (16 u32)
    cur_sub_pos: usize,
}

fn get_const(index: usize, number: u32) -> u32 {
    let b = match index {
        0 | 5 => (number >> 24) & 0xFF,
        1 => (number >> 16) & 0xFF,
        2 => (number >> 8) & 0xFF,
        3 | 4 => number & 0xFF,
        _ => unreachable!(),
    };
    CONST_1[index][b as usize]
}

fn get_const03(n: u32) -> u32 {
    get_const(0, n) ^ get_const(1, n) ^ get_const(2, n) ^ get_const(3, n)
}

// Value form of setTwoVar: returns (newSetVar, newSetVar2, newLastSum).
fn set_two_var(
    set_var: u32,
    num_fs: u32,
    num_l: u32,
    num_c: u32,
    num_p: u32,
    last_sum: u32,
) -> (u32, u32, u32) {
    let first_round = get_const03(num_fs);
    let mut sr = get_const(5, set_var) ^ get_const(4, num_l);
    sr ^= (set_var << 8) ^ (num_l >> 8) ^ num_c ^ first_round ^ last_sum;
    let nsv = sr;
    let nsv2 = num_p.wrapping_add(first_round);
    (nsv, nsv2, nsv.wrapping_add(nsv2))
}

// Value form of setFourVar: returns (svar54, svarab, svar03, svarn).
fn set_four_var(
    var5: u32,
    var4: u32,
    var_l: u32,
    var_r: u32,
    var_c: u32,
    var_a: u32,
    var_b: u32,
    var03: u32,
    var_oth: u32,
) -> (u32, u32, u32, u32) {
    let t54 = get_const(5, var5) ^ get_const(4, var4) ^ (var_l << 8) ^ var_c ^ (var_r >> 8);
    let tab = var_a.wrapping_add(var_b);
    let t03 = get_const03(var03);
    let tn = (t54.wrapping_add(tab)) ^ t03 ^ var_oth;
    (t54, tab, t03, tn)
}

impl Default for Rho5KeyProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Rho5KeyProvider {
    pub fn new() -> Self {
        Rho5KeyProvider {
            var_0: [0; 0x14],
            var_1: [0; 0x10],
            cur_sub_pos: 0,
        }
    }

    // A faithful port of the SSE2 key schedule: some `last_sum` writes mirror the
    // C# scratch aliasing through ref params and are redundant by Rust dataflow.
    #[allow(unused_assignments)]
    pub fn init_from_key(&mut self, key: &[u8]) {
        self.var_0 = [0; 0x14];
        self.var_1 = [0; 0x10];
        for i in 0..4 {
            let mut temp_num: u32 = 0;
            for j in 0..4 {
                temp_num = (temp_num << 8) | ((key[(i << 2) + j] as i8) as i32 as u32);
            }
            self.var_0[i] = temp_num;
            self.var_0[i + 0x8] = temp_num;
            self.var_0[i + 0x4] = !temp_num;
            self.var_0[i + 0x0C] = !temp_num;
        }
        let mut ta = [0u32; 0x15]; // tempArr (21 u32)
        let v = &self.var_0;
        ta[0x00 >> 2] = v[0x00 >> 2];
        ta[0x04 >> 2] = v[0x20 >> 2];
        ta[0x08 >> 2] = v[0x04 >> 2];
        ta[0x0C >> 2] = v[0x24 >> 2];
        ta[0x10 >> 2] = v[0x08 >> 2];
        ta[0x14 >> 2] = v[0x28 >> 2];
        ta[0x18 >> 2] = v[0x0C >> 2];
        ta[0x1C >> 2] = v[0x10 >> 2];
        ta[0x20 >> 2] = v[0x14 >> 2];
        ta[0x24 >> 2] = v[0x1C >> 2];
        ta[0x28 >> 2] = v[0x38 >> 2];
        ta[0x2C >> 2] = v[0x2C >> 2];
        ta[0x30 >> 2] = v[0x18 >> 2];
        ta[0x34 >> 2] = v[0x30 >> 2];
        ta[0x38 >> 2] = v[0x3C >> 2];
        ta[0x3C >> 2] = v[0x34 >> 2];
        ta[0x40 >> 2] = 0;
        ta[0x44 >> 2] = v[0x34 >> 2];
        ta[0x48 >> 2] = 0;
        ta[0x4C >> 2] = 2;

        // var_0[0x48>>2] is read/written via the spec helpers; track it explicitly.
        let mut roundcount = 2;
        loop {
            let first_round: u32 = 0;
            let mut last_sum = (ta[0x40 >> 2].wrapping_add(ta[0x00 >> 2])) ^ ta[0x48 >> 2];
            ta[0x48 >> 2] = ta[0x48 >> 2].wrapping_add(ta[0x14 >> 2]);
            let mut second_round = get_const(5, ta[0x38 >> 2]) ^ get_const(4, ta[0x1C >> 2]);
            second_round ^= (ta[0x38 >> 2] << 8)
                ^ (ta[0x1C >> 2] >> 8)
                ^ ta[0x44 >> 2]
                ^ first_round
                ^ last_sum;
            ta[0x38 >> 2] = second_round;
            last_sum = ta[0x38 >> 2].wrapping_add(ta[0x48 >> 2]);

            // The C# uses var_0[0x48>>2] (=v18) as scratch aliased through ref params.
            macro_rules! stv {
                ($sv:expr, $sv2:expr, $fs:expr, $l:expr, $c:expr, $p:expr) => {{
                    let (a, b, ls) = set_two_var($sv, $fs, $l, $c, $p, last_sum);
                    $sv = a;
                    $sv2 = b;
                    last_sum = ls;
                }};
            }
            // spec: in C# the by-value args (numFs/numL/numC/numP) are evaluated at
            // the call, i.e. BEFORE the body does `dest = source`; setVar is a ref,
            // read AFTER. Replicate that order exactly (some calls alias dest).
            macro_rules! stv_spec {
                ($sv:expr, $sv2:expr, $fs:expr, $l:expr, $c:expr, $p:expr, $dest:expr, $src:expr) => {{
                    let fs = $fs;
                    let l = $l;
                    let c = $c;
                    let p = $p;
                    $dest = $src;
                    let sv_in = $sv;
                    let (a, b, ls) = set_two_var(sv_in, fs, l, c, p, last_sum);
                    $sv = a;
                    $sv2 = b;
                    last_sum = ls;
                }};
            }

            // [ebp-28],[ebp-40]
            stv!(
                ta[0x28 >> 2],
                ta[0x40 >> 2],
                ta[0x40 >> 2],
                ta[0x18 >> 2],
                ta[0x34 >> 2],
                ta[0x0C >> 2]
            );
            // [ebp-3C],[ebp-44]
            stv!(
                ta[0x3C >> 2],
                ta[0x44 >> 2],
                ta[0x48 >> 2],
                ta[0x10 >> 2],
                ta[0x2C >> 2],
                ta[0x04 >> 2]
            );
            // ebp-34, v18
            stv!(
                ta[0x34 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x40 >> 2],
                ta[0x08 >> 2],
                ta[0x14 >> 2],
                ta[0x24 >> 2]
            );
            // ebp-2C, v18 ; dest=ta[0x48], src=v18
            stv_spec!(
                ta[0x2C >> 2],
                self.var_0[0x48 >> 2],
                ta[0x44 >> 2],
                ta[0x00 >> 2],
                ta[0x0C >> 2],
                ta[0x30 >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2]
            );
            // ebp-14, ta[0x48]
            stv!(
                ta[0x14 >> 2],
                ta[0x48 >> 2],
                ta[0x48 >> 2],
                ta[0x38 >> 2],
                ta[0x04 >> 2],
                ta[0x20 >> 2]
            );
            // ebp-0C, ta[0x48] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x0C >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x28 >> 2],
                ta[0x24 >> 2],
                ta[0x1C >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-04, ta[0x48] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x04 >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x3C >> 2],
                ta[0x30 >> 2],
                ta[0x18 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-24, ta[0x48] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x24 >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x34 >> 2],
                ta[0x20 >> 2],
                ta[0x10 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-30, ta[0x48] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x30 >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x2C >> 2],
                ta[0x1C >> 2],
                ta[0x08 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-20, ta[0x48] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x20 >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x14 >> 2],
                ta[0x18 >> 2],
                ta[0x00 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-1C, ta[0x48] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x1C >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x0C >> 2],
                ta[0x10 >> 2],
                ta[0x38 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-18, ta[0x48] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x18 >> 2],
                ta[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x04 >> 2],
                ta[0x08 >> 2],
                ta[0x28 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-10, ta[0x40] ; dest=v18, src=ta[0x48]
            stv_spec!(
                ta[0x10 >> 2],
                ta[0x40 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x24 >> 2],
                ta[0x00 >> 2],
                ta[0x3C >> 2],
                self.var_0[0x48 >> 2],
                ta[0x48 >> 2]
            );
            // ebp-08, v18
            stv!(
                ta[0x08 >> 2],
                self.var_0[0x48 >> 2],
                self.var_0[0x48 >> 2],
                ta[0x30 >> 2],
                ta[0x38 >> 2],
                ta[0x34 >> 2]
            );
            ta[0x50 >> 2] = get_const03(ta[0x40 >> 2]) ^ last_sum;
            // ebp+08, ta[0x40]
            stv!(
                ta[0x00 >> 2],
                ta[0x40 >> 2],
                ta[0x40 >> 2],
                ta[0x20 >> 2],
                ta[0x28 >> 2],
                ta[0x2C >> 2]
            );
            ta[0x48 >> 2] = get_const03(self.var_0[0x48 >> 2]);
            ta[0x4C >> 2] = ta[0x4C >> 2].wrapping_sub(1);
            self.var_0[0x48 >> 2] = ta[0x40 >> 2];
            ta[0x44 >> 2] = ta[0x3C >> 2];

            roundcount -= 1;
            if roundcount <= 0 {
                break;
            }
        }
        self.var_0[0x00 >> 2] = ta[0x00 >> 2];
        self.var_0[0x20 >> 2] = ta[0x04 >> 2];
        self.var_0[0x04 >> 2] = ta[0x08 >> 2];
        self.var_0[0x24 >> 2] = ta[0x0C >> 2];
        self.var_0[0x08 >> 2] = ta[0x10 >> 2];
        self.var_0[0x28 >> 2] = ta[0x14 >> 2];
        self.var_0[0x0C >> 2] = ta[0x18 >> 2];
        self.var_0[0x2C >> 2] = ta[0x2C >> 2];
        self.var_0[0x10 >> 2] = ta[0x1C >> 2];
        self.var_0[0x14 >> 2] = ta[0x20 >> 2];
        self.var_0[0x18 >> 2] = ta[0x30 >> 2];
        self.var_0[0x1C >> 2] = ta[0x24 >> 2];
        self.var_0[0x40 >> 2] = ta[0x50 >> 2];
        self.var_0[0x3C >> 2] = ta[0x38 >> 2];
        self.var_0[0x34 >> 2] = ta[0x3C >> 2];
        self.var_0[0x44 >> 2] = ta[0x40 >> 2];
        self.var_0[0x38 >> 2] = ta[0x28 >> 2];
        self.var_0[0x4C >> 2] = ta[0x48 >> 2];
        self.var_0[0x30 >> 2] = ta[0x34 >> 2];
        self.cur_sub_pos = 0x10;
    }

    fn refresh(&mut self) {
        for i in 0..16usize {
            let i5 = ((0x40 + 0x3C - (i << 2)) & 0x3F) >> 2;
            let i4 = ((0x40 + 0x10 - (i << 2)) & 0x3F) >> 2;
            let il = i5;
            let ic = ((0x40 + 0x34 - (i << 2)) & 0x3F) >> 2;
            let ir = i4;
            let ia = 0x4C >> 2;
            let ib = ((0x40 + 0x28 - (i << 2)) & 0x3F) >> 2;
            let i03 = 0x48 >> 2;
            let ioth = ((0x40 + 0x38 - (i << 2)) & 0x3F) >> 2;
            let (s54, sab, s03, sn) = set_four_var(
                self.var_0[i5],
                self.var_0[i4],
                self.var_0[il],
                self.var_0[ir],
                self.var_0[ic],
                self.var_0[ia],
                self.var_0[ib],
                self.var_0[i03],
                self.var_0[ioth],
            );
            self.var_0[i5] = s54; // is54 = index_5
            self.var_0[0x48 >> 2] = sab; // isab = 0x48>>2
            self.var_0[0x4C >> 2] = s03; // is03 = 0x4C>>2
            self.var_1[i] = sn;
        }
        self.cur_sub_pos = 0;
    }

    pub fn next_sub_num(&mut self) -> u32 {
        if self.cur_sub_pos == 0x10 {
            self.refresh();
        }
        let v = self.var_1[self.cur_sub_pos];
        self.cur_sub_pos += 1;
        v
    }

    pub fn init_header_key(&mut self, filename: &str, another_data: &str) {
        let arr = get_init_header_arr(filename, another_data);
        self.init_from_key(&arr);
    }

    pub fn init_files_info_key(&mut self, filename: &str, another_data: &str) {
        let arr = get_init_files_info_arr(filename, another_data);
        self.init_from_key(&arr);
    }
}

fn utf16le_bytes(s: &str) -> Vec<u8> {
    let mut b = Vec::with_capacity(s.len() * 2);
    for u in s.encode_utf16() {
        b.push((u & 0xFF) as u8);
        b.push((u >> 8) as u8);
    }
    b
}

fn get_init_header_arr(filename: &str, another_data: &str) -> [u8; 0x80] {
    let new_str = format!("{}{}", filename.to_lowercase(), another_data);
    let data = utf16le_bytes(&new_str);
    let reads_count = data.len() >> 1;
    let mut out = [0u8; 0x80];
    for i in 0..128usize {
        let index = i % reads_count;
        out[i] = (data[index * 2] as u32).wrapping_add(i as u32) as u8;
    }
    out
}

fn get_init_files_info_arr(filename: &str, another_data: &str) -> [u8; 0x80] {
    let new_str = format!("{}{}", filename.to_lowercase(), another_data);
    let data = utf16le_bytes(&new_str);
    let reads_count = data.len() >> 1;
    let str_len = new_str.encode_utf16().count();
    let mut out = [0u8; 0x80];
    for i in 0..128usize {
        let index = i % reads_count;
        // mul = i%3 + 2 (the C# magic-number division by 3)
        let mul = ((i % 3) as u32).wrapping_add(2);
        let src = data[(str_len - index - 1) * 2] as u32;
        out[i] = src.wrapping_mul(mul).wrapping_add(i as u32) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(hex: &str) -> Vec<u32> {
        hex.split_whitespace()
            .map(|s| u32::from_str_radix(s, 16).unwrap())
            .collect()
    }

    #[test]
    fn init_from_key_var0_matches_reference() {
        let mut p = Rho5KeyProvider::new();
        let key: Vec<u8> = (0u8..16).collect();
        p.init_from_key(&key);
        let expect = parse("f5c3973d b297fb6d d536ebe8 06c087ed 6f40fb05 0243cdf6 80771595 996e5817 e7e81417 ece8a954 3776b79e 48b3cb72 2ae27c2d b2840f97 cdf57916 da911aa0 a30f019a d3562ee8 d3562ee8 40e1dd87");
        assert_eq!(p.var_0.to_vec(), expect);
    }

    #[test]
    fn subnums_match_reference() {
        let mut p = Rho5KeyProvider::new();
        let key: Vec<u8> = (0u8..16).collect();
        p.init_from_key(&key);
        let got: Vec<u32> = (0..32).map(|_| p.next_sub_num()).collect();
        let expect = parse("f7cae8b2 f2bc04c3 a4566c50 116be4b0 49fd4f55 f2955b57 9e5f9c10 e18f52b2 b925173c fef415bf b6156b18 8b474280 c12de6e4 85f11ef3 1651a903 856e3847 08b42c1e ef0a0c03 4e22f885 edea5531 7553253c fa6af5c8 f73ce594 676d6829 2c907b72 b4bf520a 35319769 04d7cc88 578c1b03 f6c4a06d b26bee90 02e01a03");
        assert_eq!(got, expect);
    }

    #[test]
    fn header_key_subnums_match() {
        let mut p = Rho5KeyProvider::new();
        p.init_header_key("DataPack3_00006.rho5", "d$Bjgfc8@dH4TQ?k");
        let got: Vec<u32> = (0..16).map(|_| p.next_sub_num()).collect();
        let expect = parse("be78e83c 4b5dd2bb 2f86c6a2 829c4fb6 5e4c795f d4fe95ec 306dcfbe 125459ab 9eb0f412 c6f0bc3a 7ca54cb7 c55d3766 e0c5e983 f1caf592 1ae5117d 31a3ae02");
        assert_eq!(got, expect);
    }

    #[test]
    fn files_info_key_subnums_match() {
        let mut p = Rho5KeyProvider::new();
        p.init_files_info_key("DataPack3_00006.rho5", "d$Bjgfc8@dH4TQ?k");
        let got: Vec<u32> = (0..16).map(|_| p.next_sub_num()).collect();
        let expect = parse("5f434132 82d24140 8747dcb4 40e97885 9c79ccf7 46d6432d 7dc05a9f b0cf1aca e3acfc0a 57a50a41 f06e711c 35243ede bf843daa ae4ff532 83406a27 cd9ae44b");
        assert_eq!(got, expect);
    }
}
