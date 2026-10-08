# `.rho5` format notes (for M1c)

Research captured from KartLibrary (`Rho5`, `Rho5FileInfo`, `Rho5DecryptStream`,
`Rho5KeyProvider`, `Rho5Key`) so the Rust port can be written without re-reading
the C#. `.rho5` is the **modern** client's DataPack format; the classic client is
entirely `.rho` (already done in M1b), and `.rho5` content is also available
pre-extracted under `pipeline/unpacked/` (C#-decoded), so M1c does **not** block
the render path.

## Archive structure

Opened via `Rho5(fileName, region)`:

- **region → `anotherData`** (16-char salt mixed into every key):
  - KR `y&errfV6GRS!e8JL` · CN `d$Bjgfc8@dH4TQ?k` · TW `t5rHKg-g9BA7%=qD`
- A `Rho5DecryptStream` is opened with the **header key** (`InitHeaderKey(fileName, anotherData)`).
- `headerOffset = GetHeaderOffset(fileName)`; `fileNameOffset = headerOffset + GetFileNamesOffset(fileName)` (both in `Rho5.cs`).
- At `headerOffset`: `i32 packageHeaderCrc`, `u8 PackageVersion`, `i32 fileCount`;
  assert `packageHeaderCrc == PackageVersion + fileCount`.
- Seek `fileNameOffset`, switch stream to the **files-info key** (`InitFilesInfoKey(fileName, anotherData)`), then read `fileCount` entries:
  - `FullPath` (UTF-16 length-prefixed: `i32 len` then `len*2` bytes)
  - `i32 FileInfoChecksum`, `i32 Unknown`, `i32 Offset`, `i32 DecompressedSize`, `i32 CompressedSize`, `byte[16] Key`
- `DataBaseOffset = (streamPos + 0x3FF) >> 10 << 10` (1 KiB aligned).

## File data (`Rho5FileInfo.GetData`)

```
decryptKey = Rho5Key.GetPackedFileKey(entry.Key[16],
                                      Rho5Key.GetFileKey_U1(anotherData),  // FNV-1a
                                      entry.FullPath)
data = read CompressedSize bytes at (Offset*0x400 + DataBaseOffset)
  // first up-to-0x400 bytes are read THROUGH a Rho5DecryptStream(decryptKey),
  // the remainder is read raw, then the WHOLE buffer is decrypted again with a
  // fresh Rho5DecryptStream(decryptKey). (Faithfully replicate this two-pass quirk.)
outdata = zlib_inflate(data)   // DecompressedSize bytes
```

## Decryption primitive (`Rho5DecryptStream`)

Stream over 64-byte blocks; per block, for each `u32` word:
`word -= keyProvider.GetNextSubNum()` (subtraction, with borrow — not XOR).
Last partial block: process `(len+3)/4` words; only the real bytes are emitted,
so the low bytes are correct even though a partial final word reads padding.

## Key derivation — the two hard pieces

- **`Rho5KeyProvider`** (~775 lines): a custom block-cipher **key schedule** +
  `GetNextSubNum()` keystream. Pure scalar (uint arrays `var_0[0x14]`,
  `var_1[0x10]`, `tempArr[0x15]`, multi-round `firstRound/secondRound` mixing).
  Three inits: `InitFromKey(byte[])`, `InitHeaderKey(name, salt)`,
  `InitFilesInfoKey(name, salt)`. Direct, if tedious, Rust translation.
- **`Rho5Key`** (~707 lines, **68 SSE2 intrinsics**): the key-derivation math
  (`GetPackedFileKey`, header/files-info key builders). The SIMD is plain
  arithmetic — `Multiply/Add/AddScalar/MultiplyScalar/ConvertToVector/UnpackLow`
  on integer **and double** lanes — **no AES**. So it's scalarizable, but each of
  the 68 sites must be hand-translated and (for the `double` lanes) kept
  bit-exact to IEEE-754. This is the reason the C# build needs x64/Rosetta.

## Port plan (M1c)

1. `Rho5KeyProvider` (scalar) + unit test `GetNextSubNum` sequence vs C#.
2. `Rho5Key` — scalarize the SSE2 math; test `GetFileKey_U1` (FNV-1a) and
   `GetPackedFileKey` against reference vectors.
3. `Rho5` archive parse + `GetData` (two-pass decrypt + zlib).
4. Validate byte-for-byte vs the C# RhoUnpacker (x64) on several `.rho5`, as M1b did.

Estimated ~1500 lines of careful translation; the SSE2→scalar pass with exact
double-precision parity is the main risk and is best done as its own focused step.
