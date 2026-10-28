# Blind-batch report — live pipeline vs. Training labels

Batch size: 154 (limit 154)  ·  splits: valid + test  ·  detection: adaptive auto mode (conf 0.1, floor off)  ·  match IoU ≥ 0.3  ·  generated 2026-10-28 12:50

## Aggregates

| metric | value |
|---|---|
| images | 154 ok / 0 errored |
| mean detection recall (images with labels) | 68% over 152 |
| images with zero detections | 0 |
| matched bananas | 231 |
| ripeness agreement | 132 / 231 (57%) |
| spurious detections (no label overlap) | 615 |

## Confusion (label row × predicted col)

| label \ pred | overripe | ripe | semi-ripe | underripe | unlabelled |
|---|---|---|---|---|---|
| overripe | 1 | 6 | 0 | 0 | 0 |
| ripe | 3 | 85 | 0 | 4 | 0 |
| underripe | 5 | 81 | 12 | 34 | 0 |
| unlabelled | 0 | 0 | 0 | 0 | 0 |

## Mismatches (matched banana whose class disagrees)

| image | # | label | pred | G% | Y% | B% | det conf |
|---|---|---|---|---|---|---|---|
| bunch_643_jpg.rf.67f35b648ec4d | 4 | overripe | ripe | 0.1 | 81.9 | 16.4 | 0.32 |
| bunch_643_jpg.rf.67f35b648ec4d | 1 | overripe | ripe | 0.0 | 55.7 | 0.4 | 0.40 |
| bunch_647_jpg.rf.e9067b006cf5c | 1 | overripe | ripe | 0.0 | 34.8 | 1.0 | 0.21 |
| bunch_647_jpg.rf.e9067b006cf5c | 2 | overripe | ripe | 0.0 | 7.4 | 0.0 | 0.16 |
| bunch_232_jpg.rf.56806cb7233e3 | 3 | underripe | ripe | 0.1 | 87.2 | 12.6 | 0.33 |
| bunch_327_jpg.rf.d8be7153a9ef3 | 2 | underripe | ripe | 3.9 | 69.6 | 2.1 | 0.21 |
| bunch_125_jpg.rf.4671325a15d80 | 2 | underripe | ripe | 1.9 | 85.8 | 3.3 | 0.30 |
| bunch_165_jpg.rf.73392960260d4 | 1 | underripe | overripe | 28.0 | 43.0 | 20.3 | 0.77 |
| bunch_175_jpg.rf.4f8a43b7e0438 | 4 | underripe | ripe | 0.1 | 80.8 | 18.6 | 0.11 |
| bunch_181_jpg.rf.be7e5040b5e06 | 2 | underripe | ripe | 16.4 | 78.6 | 2.9 | 0.71 |
| bunch_181_jpg.rf.be7e5040b5e06 | 1 | underripe | ripe | 9.6 | 73.3 | 5.0 | 0.38 |
| bunch_200_jpg.rf.68d01446a8d6e | 2 | underripe | ripe | 2.1 | 86.3 | 2.7 | 0.73 |
| bunch_200_jpg.rf.68d01446a8d6e | 1 | underripe | ripe | 4.4 | 72.7 | 9.7 | 0.28 |
| bunch_202_jpg.rf.bbcd851d25c8d | 1 | underripe | ripe | 0.0 | 87.4 | 11.6 | 0.41 |
| bunch_205_jpg.rf.330f12f1dff03 | 1 | underripe | ripe | 0.1 | 89.4 | 10.1 | 0.28 |
| bunch_208_jpg.rf.5a86923127bb7 | 1 | underripe | ripe | 0.7 | 88.1 | 5.8 | 0.22 |
| bunch_210_jpg.rf.d03f44cb4087d | 1 | underripe | ripe | 7.0 | 83.6 | 5.4 | 0.72 |
| bunch_210_jpg.rf.d03f44cb4087d | 2 | underripe | ripe | 0.2 | 96.4 | 1.8 | 0.38 |
| bunch_224_jpg.rf.e3c35a3895bda | 3 | underripe | ripe | 0.1 | 88.6 | 10.7 | 0.42 |
| bunch_225_jpg.rf.43f1993705592 | 1 | underripe | ripe | 0.0 | 98.8 | 0.9 | 0.54 |
| bunch_225_jpg.rf.43f1993705592 | 2 | underripe | ripe | 0.0 | 94.3 | 5.5 | 0.23 |
| bunch_227_jpg.rf.cf4c628ca7490 | 1 | underripe | ripe | 0.1 | 98.3 | 1.1 | 0.28 |
| bunch_252_jpg.rf.e5630742c2ae9 | 4 | underripe | ripe | 8.7 | 85.3 | 5.8 | 0.19 |
| bunch_252_jpg.rf.e5630742c2ae9 | 2 | underripe | ripe | 6.6 | 85.7 | 6.3 | 0.24 |
| bunch_252_jpg.rf.e5630742c2ae9 | 8 | underripe | overripe | 3.9 | 50.0 | 37.7 | 0.22 |
| bunch_253_jpg.rf.a9aa2b9ac03c5 | 4 | underripe | ripe | 0.3 | 83.9 | 15.2 | 0.91 |
| bunch_253_jpg.rf.a9aa2b9ac03c5 | 6 | underripe | ripe | 8.0 | 67.6 | 4.5 | 0.45 |
| bunch_273_jpg.rf.80fb5d16fee07 | 1 | underripe | ripe | 0.3 | 97.8 | 1.2 | 0.57 |
| bunch_295_jpg.rf.2c2127a1284c7 | 2 | underripe | ripe | 1.2 | 48.3 | 13.7 | 0.17 |
| bunch_307_jpg.rf.f896be506218c | 2 | underripe | ripe | 0.1 | 97.3 | 2.4 | 0.45 |
| bunch_392_jpg.rf.fe5acd1a3ea05 | 4 | underripe | ripe | 19.6 | 71.5 | 7.8 | 0.37 |
| bunch_392_jpg.rf.fe5acd1a3ea05 | 1 | underripe | ripe | 0.6 | 79.7 | 19.5 | 0.77 |
| bunch_444_jpg.rf.c77d1c225eca9 | 1 | underripe | ripe | 0.4 | 89.0 | 8.5 | 0.56 |
| bunch_445_jpg.rf.c22643daff6c9 | 1 | underripe | ripe | 0.0 | 91.5 | 6.6 | 0.81 |
| bunch_451_jpg.rf.46ddcb8ce13e7 | 1 | underripe | ripe | 0.0 | 92.6 | 6.8 | 0.25 |
| bunch_454_jpg.rf.d9602b659c8c7 | 1 | underripe | ripe | 0.9 | 88.5 | 7.7 | 0.43 |
| bunch_459_jpg.rf.e49b8d92e05b1 | 3 | underripe | ripe | 0.2 | 97.9 | 1.8 | 0.28 |
| bunch_45_jpg.rf.ded71a67a7b9ff | 1 | underripe | ripe | 0.5 | 97.1 | 2.3 | 0.61 |
| bunch_46_jpg.rf.c67a13a7a11f47 | 3 | underripe | ripe | 0.0 | 99.1 | 0.0 | 0.22 |
| bunch_508_jpg.rf.1bf58f1f85493 | 1 | underripe | ripe | 0.2 | 88.2 | 11.4 | 0.59 |
| bunch_509_jpg.rf.17751229d96d1 | 1 | underripe | ripe | 0.1 | 94.0 | 5.5 | 0.22 |
| bunch_521_jpg.rf.9eb6fab1cdd1c | 1 | underripe | ripe | 0.2 | 99.8 | 0.0 | 0.61 |
| bunch_571_jpg.rf.5f738bda47ba1 | 1 | underripe | ripe | 0.0 | 99.5 | 0.3 | 0.88 |
| bunch_608_jpg.rf.7490bd7bb1932 | 1 | underripe | ripe | 0.3 | 87.9 | 11.3 | 0.36 |
| bunch_65_jpg.rf.3797af5fe0ed5f | 3 | underripe | ripe | 15.7 | 83.9 | 0.2 | 0.36 |
| bunch_65_jpg.rf.3797af5fe0ed5f | 2 | underripe | ripe | 0.3 | 86.9 | 11.5 | 0.32 |
| bunch_440_jpg.rf.9ae90b20e93d4 | 1 | overripe | ripe | 0.1 | 52.0 | 7.1 | 0.18 |
| bunch_185_jpg.rf.b621f38f16a29 | 1 | underripe | ripe | 0.2 | 86.6 | 4.9 | 0.35 |
| bunch_26_jpg.rf.519721a7f222b7 | 1 | ripe | overripe | 0.1 | 64.6 | 34.1 | 0.41 |
| bunch_340_jpg.rf.1cf347b36f4b7 | 2 | ripe | overripe | 0.1 | 69.0 | 20.5 | 0.41 |
| bunch_348_jpg.rf.160173c7f65ef | 1 | ripe | overripe | 0.0 | 74.0 | 24.0 | 0.20 |
| bunch_465_jpg.rf.b4d8dca50180a | 5 | underripe | ripe | 0.0 | 92.0 | 0.0 | 0.14 |
| bunch_112_jpg.rf.6b61510258457 | 4 | ripe | unripe | 0.0 | 0.0 | 0.0 | 0.37 |
| bunch_167_jpg.rf.fae2d400e1fe7 | 4 | underripe | ripe | 0.4 | 88.0 | 11.2 | 0.42 |
| bunch_171_jpg.rf.7d55d6339b026 | 5 | underripe | ripe | 1.5 | 77.9 | 8.8 | 0.12 |
| bunch_171_jpg.rf.7d55d6339b026 | 4 | underripe | ripe | 0.0 | 85.4 | 6.5 | 0.19 |
| bunch_171_jpg.rf.7d55d6339b026 | 3 | underripe | ripe | 12.0 | 79.1 | 0.8 | 0.19 |
| bunch_197_jpg.rf.997be390a5c7b | 1 | underripe | ripe | 2.4 | 86.7 | 4.3 | 0.67 |
| bunch_236_jpg.rf.df498500f60b3 | 1 | underripe | ripe | 0.0 | 98.5 | 1.3 | 0.36 |
| bunch_236_jpg.rf.df498500f60b3 | 2 | underripe | ripe | 0.0 | 93.4 | 6.6 | 0.13 |
| bunch_241_jpg.rf.f8fe539bb1abf | 1 | underripe | ripe | 0.4 | 80.2 | 13.1 | 0.50 |
| bunch_241_jpg.rf.f8fe539bb1abf | 3 | underripe | ripe | 0.1 | 95.6 | 4.0 | 0.72 |
| bunch_246_jpg.rf.937d8a75eaba0 | 1 | underripe | ripe | 1.9 | 40.0 | 16.1 | 0.45 |
| bunch_256_jpg.rf.c5eef63f143a9 | 4 | underripe | ripe | 6.1 | 74.0 | 15.4 | 0.15 |
| bunch_256_jpg.rf.c5eef63f143a9 | 3 | underripe | overripe | 11.4 | 45.3 | 40.0 | 0.16 |
| bunch_260_jpg.rf.670bb5f5ba073 | 1 | ripe | unripe | 0.0 | 0.0 | 0.0 | 0.83 |
| bunch_260_jpg.rf.670bb5f5ba073 | 2 | underripe | ripe | 9.7 | 47.2 | 0.4 | 0.57 |
| bunch_267_jpg.rf.1eaefd3d149e5 | 1 | underripe | ripe | 0.2 | 93.8 | 5.7 | 0.56 |
| bunch_267_jpg.rf.1eaefd3d149e5 | 2 | ripe | unripe | 31.2 | 23.8 | 0.0 | 0.26 |
| bunch_291_jpg.rf.e5dca3811156e | 2 | underripe | ripe | 0.2 | 97.2 | 1.9 | 0.19 |
| bunch_291_jpg.rf.e5dca3811156e | 1 | ripe | unripe | 0.0 | 0.0 | 0.0 | 0.47 |
| bunch_304_jpg.rf.a3dd6998a5c6d | 2 | overripe | ripe | 1.0 | 28.2 | 18.9 | 0.14 |
| bunch_304_jpg.rf.a3dd6998a5c6d | 1 | underripe | overripe | 1.0 | 75.3 | 20.2 | 0.15 |
| bunch_304_jpg.rf.a3dd6998a5c6d | 3 | underripe | ripe | 2.8 | 75.1 | 16.3 | 0.38 |
| bunch_309_jpg.rf.634a3be8a108d | 4 | underripe | ripe | 19.0 | 67.4 | 0.1 | 0.18 |
| bunch_350_jpg.rf.030c1c50dc2ee | 2 | underripe | ripe | 1.1 | 64.0 | 2.4 | 0.50 |
| bunch_359_jpg.rf.04c266ec33ac9 | 5 | underripe | ripe | 2.1 | 83.1 | 9.3 | 0.32 |
| bunch_359_jpg.rf.04c266ec33ac9 | 3 | underripe | overripe | 22.6 | 23.7 | 33.9 | 0.30 |
| bunch_359_jpg.rf.04c266ec33ac9 | 1 | underripe | ripe | 1.0 | 87.0 | 9.5 | 0.35 |
| bunch_395_jpg.rf.7244f7360becf | 1 | underripe | ripe | 0.8 | 91.5 | 7.5 | 0.58 |
| bunch_42_jpg.rf.517a873a24fbb0 | 6 | underripe | ripe | 16.9 | 72.5 | 1.4 | 0.66 |
| bunch_452_jpg.rf.c7855cb95a26c | 1 | underripe | ripe | 0.0 | 99.3 | 0.6 | 0.31 |
| bunch_474_jpg.rf.63445b0e61e23 | 1 | underripe | ripe | 0.0 | 94.3 | 5.4 | 0.66 |
| bunch_475_jpg.rf.ea005c7f50ced | 1 | underripe | ripe | 1.7 | 78.4 | 12.4 | 0.14 |
| bunch_482_jpg.rf.032a1fe534c97 | 1 | underripe | ripe | 3.5 | 85.4 | 8.8 | 0.80 |
| bunch_48_jpg.rf.e0ea423ca4aa11 | 1 | underripe | ripe | 3.7 | 80.2 | 4.2 | 0.46 |
| bunch_50_jpg.rf.d773a3e466f02c | 1 | underripe | ripe | 11.9 | 80.8 | 2.0 | 0.30 |
| bunch_544_jpg.rf.77d9672ecbc89 | 1 | underripe | ripe | 0.1 | 98.7 | 1.1 | 0.69 |
| bunch_559_jpg.rf.c4ba31e84dedf | 1 | underripe | ripe | 15.2 | 79.6 | 2.6 | 0.39 |
| bunch_575_jpg.rf.cb0225d1f2d39 | 1 | underripe | ripe | 6.3 | 81.8 | 7.4 | 0.28 |
| bunch_593_jpg.rf.facfe498c8339 | 1 | underripe | ripe | 10.2 | 86.5 | 0.9 | 0.34 |
| bunch_611_jpg.rf.131d60f40ec7f | 1 | underripe | ripe | 0.2 | 79.2 | 19.7 | 0.14 |
| bunch_78_jpg.rf.555d926aa73ea2 | 1 | underripe | ripe | 0.0 | 88.5 | 10.9 | 0.63 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 1 | underripe | ripe | 0.2 | 86.9 | 4.9 | 0.56 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 2 | underripe | ripe | 0.1 | 96.8 | 3.0 | 0.76 |
| bunch_82_jpg.rf.dea869305eddf2 | 1 | underripe | ripe | 2.4 | 70.8 | 3.0 | 0.44 |
| bunch_92_jpg.rf.a004bc12cf3c06 | 1 | underripe | ripe | 1.3 | 22.4 | 0.7 | 0.23 |
| bunch_97_jpg.rf.0ee9408bb835d7 | 1 | underripe | ripe | 16.3 | 72.7 | 0.0 | 0.28 |
| bunch_97_jpg.rf.0ee9408bb835d7 | 4 | underripe | ripe | 2.2 | 44.5 | 1.7 | 0.57 |

## Per-image detail

| image | labels | dets | matched | recall | spurious | sec |
|---|---|---|---|---|---|---|
| bunch_100_jpg.rf.a1b2603b38385cde1eb1782 | 1 | 2 | 0 | 0% | 2 | 5.52 |
| bunch_643_jpg.rf.67f35b648ec4dfd74dbfae1 | 6 | 3 | 2 | 33% | 1 | 1.13 |
| bunch_647_jpg.rf.e9067b006cf5ca75b95818c | 4 | 10 | 3 | 75% | 7 | 1.0 |
| bunch_120_jpg.rf.5cc9f4e5b3eca6edbd3611a | 2 | 6 | 0 | 0% | 6 | 0.58 |
| bunch_137_jpg.rf.afdb40dea51966e920b1427 | 7 | 9 | 1 | 14% | 8 | 0.9 |
| bunch_147_jpg.rf.ab64917eab7cfa2db5b3852 | 3 | 17 | 2 | 67% | 15 | 1.2 |
| bunch_232_jpg.rf.56806cb7233e30f1c313d6a | 3 | 7 | 3 | 100% | 4 | 0.5 |
| bunch_23_jpg.rf.62b7b717609d9bae9a083523 | 1 | 1 | 1 | 100% | 0 | 0.79 |
| bunch_289_jpg.rf.ebba34640b3e54eacc6afa0 | 3 | 4 | 1 | 33% | 3 | 0.43 |
| bunch_290_jpg.rf.5bab7360124fe3fde2cd6ed | 4 | 10 | 3 | 75% | 7 | 1.04 |
| bunch_292_jpg.rf.03ff4bdea42c1e84dcbc902 | 2 | 8 | 2 | 100% | 6 | 1.18 |
| bunch_302_jpg.rf.c9b5780d914946c6e79f1f9 | 8 | 7 | 3 | 38% | 4 | 0.92 |
| bunch_317_jpg.rf.f0bf749b2019f1d4447fa19 | 5 | 5 | 2 | 40% | 3 | 0.96 |
| bunch_327_jpg.rf.d8be7153a9ef3a49a86c3fa | 3 | 3 | 2 | 67% | 1 | 0.62 |
| bunch_329_jpg.rf.630135c02b4c569025e27ba | 8 | 26 | 7 | 88% | 19 | 1.16 |
| bunch_372_jpg.rf.1c13957182bd1fb9f774953 | 1 | 6 | 1 | 100% | 5 | 0.92 |
| bunch_374_jpg.rf.8a293009a0a102e6d504ee8 | 2 | 4 | 1 | 50% | 3 | 0.61 |
| bunch_419_jpg.rf.d7ab8908bed8a7954446f21 | 1 | 4 | 1 | 100% | 3 | 0.58 |
| bunch_423_jpg.rf.1908ae9c3bf01ef4fd8a6a5 | 1 | 4 | 1 | 100% | 3 | 1.3 |
| bunch_427_jpg.rf.8aa3be5427be0c48314f836 | 1 | 4 | 1 | 100% | 3 | 0.78 |
| bunch_430_jpg.rf.2ed80d4fbecdb13ff3af2d0 | 1 | 2 | 0 | 0% | 2 | 0.99 |
| bunch_432_jpg.rf.5a0099d014b0321991e3188 | 1 | 8 | 1 | 100% | 7 | 1.76 |
| bunch_617_jpg.rf.306612ef2ad9e6208969938 | 1 | 7 | 1 | 100% | 6 | 1.0 |
| bunch_80_jpg.rf.ca576b9f9efb0a1438b80c64 | 8 | 4 | 3 | 38% | 1 | 1.17 |
| bunch_87_jpg.rf.33cb34b7da304b1a98f3905b | 1 | 3 | 0 | 0% | 3 | 0.57 |
| bunch_125_jpg.rf.4671325a15d8078b7f5ca10 | 2 | 5 | 1 | 50% | 4 | 0.32 |
| bunch_134_jpg.rf.7d66b62763ff7e210845470 | 6 | 9 | 1 | 17% | 8 | 0.81 |
| bunch_165_jpg.rf.73392960260d47ff5fc58a4 | 3 | 2 | 1 | 33% | 1 | 0.5 |
| bunch_175_jpg.rf.4f8a43b7e0438bbdfcf338f | 4 | 5 | 2 | 50% | 3 | 0.59 |
| bunch_181_jpg.rf.be7e5040b5e06142d6a1120 | 3 | 3 | 3 | 100% | 0 | 0.54 |
| bunch_184_jpg.rf.e6f1cc8448fbc5de2213b48 | 3 | 5 | 2 | 67% | 3 | 0.78 |
| bunch_187_jpg.rf.847fa6f325b41affd82f049 | 2 | 7 | 2 | 100% | 5 | 1.07 |
| bunch_200_jpg.rf.68d01446a8d6ef7180b7541 | 2 | 8 | 2 | 100% | 6 | 0.81 |
| bunch_202_jpg.rf.bbcd851d25c8d78fed28b3f | 2 | 5 | 2 | 100% | 3 | 0.45 |
| bunch_205_jpg.rf.330f12f1dff03d0fcedcd0f | 4 | 2 | 1 | 25% | 1 | 0.46 |
| bunch_208_jpg.rf.5a86923127bb7ef96bd9141 | 2 | 1 | 1 | 50% | 0 | 0.33 |
| bunch_210_jpg.rf.d03f44cb4087d79f8cf5d64 | 3 | 2 | 2 | 67% | 0 | 0.36 |
| bunch_214_jpg.rf.24ae57a0f2e7fee73bdf2bc | 2 | 8 | 1 | 50% | 7 | 0.57 |
| bunch_224_jpg.rf.e3c35a3895bda9717a4bbb9 | 3 | 6 | 2 | 67% | 4 | 0.72 |
| bunch_225_jpg.rf.43f19937055926a72ba72af | 2 | 11 | 2 | 100% | 9 | 1.26 |
| bunch_227_jpg.rf.cf4c628ca749030e63b2cd3 | 1 | 9 | 1 | 100% | 8 | 0.79 |
| bunch_252_jpg.rf.e5630742c2ae9f155c9fb0e | 8 | 5 | 3 | 38% | 2 | 0.57 |
| bunch_253_jpg.rf.a9aa2b9ac03c5fcc5e8009a | 7 | 2 | 2 | 29% | 0 | 0.85 |
| bunch_273_jpg.rf.80fb5d16fee0753d30787ce | 1 | 5 | 1 | 100% | 4 | 1.0 |
| bunch_295_jpg.rf.2c2127a1284c71f64be4054 | 3 | 12 | 2 | 67% | 10 | 1.3 |
| bunch_307_jpg.rf.f896be506218c7f026cdd5e | 5 | 7 | 2 | 40% | 5 | 0.7 |
| bunch_313_jpg.rf.462dad60241ac00dcac6960 | 3 | 4 | 2 | 67% | 2 | 0.86 |
| bunch_387_jpg.rf.8693a9674f78ebb1415dc98 | 3 | 7 | 2 | 67% | 5 | 1.02 |
| bunch_392_jpg.rf.fe5acd1a3ea05e88f65b1ef | 4 | 3 | 2 | 50% | 1 | 1.18 |
| bunch_444_jpg.rf.c77d1c225eca9275b2ca413 | 1 | 5 | 1 | 100% | 4 | 0.77 |
| bunch_445_jpg.rf.c22643daff6c95bd477458b | 1 | 1 | 1 | 100% | 0 | 0.35 |
| bunch_451_jpg.rf.46ddcb8ce13e73d3bcb4b65 | 1 | 8 | 1 | 100% | 7 | 1.39 |
| bunch_454_jpg.rf.d9602b659c8c77bf5940134 | 1 | 7 | 1 | 100% | 6 | 0.89 |
| bunch_459_jpg.rf.e49b8d92e05b1e2b08ed2a7 | 4 | 7 | 2 | 50% | 5 | 0.77 |
| bunch_45_jpg.rf.ded71a67a7b9ff84df88a891 | 4 | 7 | 3 | 75% | 4 | 0.27 |
| bunch_462_jpg.rf.49bcee60c02ad4156a6a7ba | 5 | 9 | 1 | 20% | 8 | 2.21 |
| bunch_466_jpg.rf.d6722658d41529496258d47 | 5 | 2 | 1 | 20% | 1 | 1.21 |
| bunch_46_jpg.rf.c67a13a7a11f47070f32f45f | 3 | 2 | 1 | 33% | 1 | 0.31 |
| bunch_508_jpg.rf.1bf58f1f8549386aa36872f | 1 | 3 | 1 | 100% | 2 | 0.64 |
| bunch_509_jpg.rf.17751229d96d19bbcf1d55e | 1 | 5 | 1 | 100% | 4 | 1.13 |
| bunch_516_jpg.rf.49a31f9cd5a23f8f3de0315 | 1 | 1 | 1 | 100% | 0 | 0.66 |
| bunch_519_jpg.rf.dabeda794cb942c0866c8f1 | 1 | 2 | 1 | 100% | 1 | 0.93 |
| bunch_520_jpg.rf.6217bf94a1004735a44ade8 | 1 | 3 | 1 | 100% | 2 | 0.77 |
| bunch_521_jpg.rf.9eb6fab1cdd1c4b4f8c95bd | 1 | 3 | 1 | 100% | 2 | 1.19 |
| bunch_525_jpg.rf.d9a175de13db2946f74a0ea | 1 | 1 | 1 | 100% | 0 | 0.46 |
| bunch_556_jpg.rf.060134afaaff5e1a222901d | 1 | 2 | 1 | 100% | 1 | 0.62 |
| bunch_55_jpg.rf.02611d3d6408d695dd2d0bd6 | 3 | 1 | 1 | 33% | 0 | 0.24 |
| bunch_569_jpg.rf.90c1d2a7f72d6d8ed9c6848 | 1 | 1 | 1 | 100% | 0 | 0.66 |
| bunch_571_jpg.rf.5f738bda47ba13b5fb1c271 | 1 | 2 | 1 | 100% | 1 | 0.68 |
| bunch_596_jpg.rf.351ddaf6d73fda837e51fdd | 1 | 2 | 0 | 0% | 2 | 0.4 |
| bunch_598_jpg.rf.13cfe98a228f6ea56795f12 | 1 | 3 | 1 | 100% | 2 | 0.74 |
| bunch_608_jpg.rf.7490bd7bb193256bdc6f7df | 1 | 6 | 1 | 100% | 5 | 1.64 |
| bunch_633_jpeg.rf.c0d7e698897ba31c9d3209 | 4 | 2 | 2 | 50% | 0 | 0.24 |
| bunch_634_jpeg.rf.d65fd2ab2655ea500d6937 | 2 | 1 | 1 | 50% | 0 | 0.24 |
| bunch_65_jpg.rf.3797af5fe0ed5f1317ec0d93 | 3 | 9 | 3 | 100% | 6 | 0.91 |
| null_33_jpg.rf.96935df24891f67395b37f95d | 0 | 2 | 0 | — | 2 | 0.15 |
| null_49_jpg.rf.b484eec1c500cc7b4c8687492 | 0 | 9 | 0 | — | 9 | 0.6 |
| bunch_426_jpg.rf.63d4554df0df892a4a1b968 | 1 | 1 | 1 | 100% | 0 | 0.9 |
| bunch_440_jpg.rf.9ae90b20e93d48a5599b6aa | 1 | 1 | 1 | 100% | 0 | 0.36 |
| bunch_618_jpg.rf.3dc1311a5cf991afaa76c5d | 1 | 4 | 0 | 0% | 4 | 0.76 |
| bunch_84_jpg.rf.2a87ae6c1e86623b3ba1ca67 | 1 | 3 | 0 | 0% | 3 | 0.5 |
| bunch_101_jpg.rf.d1a1e7823b929082e9a9f4f | 3 | 1 | 1 | 33% | 0 | 0.95 |
| bunch_111_jpg.rf.51b36f6d39aaab77633ae01 | 3 | 6 | 2 | 67% | 4 | 1.19 |
| bunch_116_jpg.rf.9f4119431a28dd43a2fc8fa | 6 | 14 | 1 | 17% | 13 | 0.95 |
| bunch_123_jpg.rf.ca8772a96f82a4b736f550a | 2 | 5 | 2 | 100% | 3 | 1.18 |
| bunch_127_jpg.rf.af196dbf72d768b89914fa2 | 5 | 2 | 2 | 40% | 0 | 0.27 |
| bunch_14_jpg.rf.e403f859d62c37e09bc493f7 | 4 | 7 | 1 | 25% | 6 | 0.59 |
| bunch_154_jpg.rf.1a1e0ebdb6f1aff6584d0ec | 2 | 4 | 0 | 0% | 4 | 0.39 |
| bunch_183_jpg.rf.27b9110a9906eada0046429 | 3 | 3 | 1 | 33% | 2 | 1.01 |
| bunch_185_jpg.rf.b621f38f16a295aebbd47e2 | 3 | 9 | 2 | 67% | 7 | 0.96 |
| bunch_24_jpg.rf.69a15577015a4c892d889e6c | 3 | 9 | 1 | 33% | 8 | 2.06 |
| bunch_26_jpg.rf.519721a7f222b756fcc53eeb | 5 | 7 | 1 | 20% | 6 | 1.73 |
| bunch_277_jpg.rf.3d0db1099ae4b31f2a81d91 | 1 | 3 | 1 | 100% | 2 | 0.43 |
| bunch_278_jpg.rf.04f2d116273c5b4c19a0c2d | 2 | 4 | 1 | 50% | 3 | 0.97 |
| bunch_27_jpg.rf.7b66ec71ac176fc7a857b19b | 4 | 13 | 2 | 50% | 11 | 1.23 |
| bunch_280_jpg.rf.81f558727fc9e3416796e0b | 1 | 3 | 1 | 100% | 2 | 0.64 |
| bunch_294_jpg.rf.80e68e3ed0ad201a58005ed | 4 | 2 | 1 | 25% | 1 | 0.46 |
| bunch_340_jpg.rf.1cf347b36f4b7268be775aa | 5 | 10 | 2 | 40% | 8 | 0.86 |
| bunch_348_jpg.rf.160173c7f65efe746b4ad7a | 8 | 15 | 3 | 38% | 12 | 0.62 |
| bunch_360_jpg.rf.652a6a7ef71e74f52af3f58 | 4 | 10 | 1 | 25% | 9 | 0.47 |
| bunch_377_jpg.rf.9b55bd2448ef3f6ca814f5a | 3 | 1 | 1 | 33% | 0 | 0.61 |
| bunch_385_jpg.rf.328b84f40d4ee1f6780c1c6 | 1 | 5 | 1 | 100% | 4 | 0.8 |
| bunch_38_jpg.rf.26d0f72f51d3b1faaa1f1bcc | 6 | 14 | 3 | 50% | 11 | 0.95 |
| bunch_401_jpg.rf.08b91df5d734ffa78ab1237 | 9 | 4 | 2 | 22% | 2 | 0.5 |
| bunch_417_jpg.rf.cf83da72e653889888810b9 | 1 | 11 | 1 | 100% | 10 | 0.72 |
| bunch_465_jpg.rf.b4d8dca50180a5ccf8262c2 | 6 | 11 | 4 | 67% | 7 | 0.91 |
| bunch_471_jpg.rf.24bb890b1d5e8dcce658c95 | 7 | 7 | 1 | 14% | 6 | 0.73 |
| bunch_47_jpg.rf.e2f2b5d445a28dd710d4dbe5 | 3 | 5 | 3 | 100% | 2 | 0.33 |
| bunch_68_jpg.rf.f3b075eeb5816019c0fb035b | 5 | 12 | 2 | 40% | 10 | 1.11 |
| bunch_69_jpg.rf.3ead8efa0a292abb523b318a | 3 | 12 | 3 | 100% | 9 | 0.87 |
| bunch_112_jpg.rf.6b615102584573c3eb2958b | 4 | 9 | 2 | 50% | 7 | 0.95 |
| bunch_167_jpg.rf.fae2d400e1fe7eed040be78 | 6 | 8 | 3 | 50% | 5 | 0.47 |
| bunch_171_jpg.rf.7d55d6339b026093a56ed1b | 5 | 4 | 4 | 80% | 0 | 0.66 |
| bunch_197_jpg.rf.997be390a5c7bc59bcb96ae | 1 | 2 | 1 | 100% | 1 | 0.4 |
| bunch_236_jpg.rf.df498500f60b3cc0533a645 | 3 | 6 | 2 | 67% | 4 | 0.96 |
| bunch_241_jpg.rf.f8fe539bb1abff8b221e71c | 3 | 5 | 2 | 67% | 3 | 0.73 |
| bunch_246_jpg.rf.937d8a75eaba0bc8f250539 | 1 | 2 | 1 | 100% | 1 | 0.42 |
| bunch_256_jpg.rf.c5eef63f143a94c5af914dd | 4 | 12 | 2 | 50% | 10 | 0.44 |
| bunch_260_jpg.rf.670bb5f5ba073621b18fd12 | 2 | 4 | 2 | 100% | 2 | 0.5 |
| bunch_267_jpg.rf.1eaefd3d149e5c1f101d31e | 2 | 7 | 2 | 100% | 5 | 0.82 |
| bunch_291_jpg.rf.e5dca3811156e90426e85a1 | 2 | 11 | 2 | 100% | 9 | 0.77 |
| bunch_304_jpg.rf.a3dd6998a5c6d17776679aa | 4 | 12 | 3 | 75% | 9 | 0.85 |
| bunch_309_jpg.rf.634a3be8a108de97d127163 | 4 | 7 | 2 | 50% | 5 | 0.86 |
| bunch_350_jpg.rf.030c1c50dc2ee956cd145da | 4 | 5 | 2 | 50% | 3 | 0.65 |
| bunch_359_jpg.rf.04c266ec33ac97a51122ecc | 7 | 5 | 4 | 57% | 1 | 0.41 |
| bunch_395_jpg.rf.7244f7360becf46248e97c7 | 2 | 7 | 1 | 50% | 6 | 0.92 |
| bunch_396_jpg.rf.f710a7bc921353787a05f28 | 2 | 1 | 1 | 50% | 0 | 0.53 |
| bunch_398_jpg.rf.a64a59303d20f8a7935d9ad | 1 | 5 | 1 | 100% | 4 | 1.21 |
| bunch_42_jpg.rf.517a873a24fbb0d2c9d1298b | 7 | 1 | 1 | 14% | 0 | 0.23 |
| bunch_44_jpg.rf.fa217c553450b1450863a9ef | 1 | 1 | 1 | 100% | 0 | 0.3 |
| bunch_452_jpg.rf.c7855cb95a26cb93abc8b8c | 1 | 2 | 1 | 100% | 1 | 0.43 |
| bunch_474_jpg.rf.63445b0e61e2372be30c7d5 | 1 | 3 | 1 | 100% | 2 | 0.71 |
| bunch_475_jpg.rf.ea005c7f50ced29d8a3f67e | 1 | 8 | 1 | 100% | 7 | 0.97 |
| bunch_482_jpg.rf.032a1fe534c9763bd29f38b | 1 | 5 | 1 | 100% | 4 | 0.53 |
| bunch_48_jpg.rf.e0ea423ca4aa114124de2151 | 1 | 3 | 1 | 100% | 2 | 0.21 |
| bunch_49_jpg.rf.2ecd888d042b512fc58c28d9 | 2 | 1 | 1 | 50% | 0 | 0.21 |
| bunch_50_jpg.rf.d773a3e466f02c060b1d6c5c | 2 | 8 | 1 | 50% | 7 | 0.22 |
| bunch_517_jpg.rf.87bb8c04473353edd7f9ec4 | 1 | 2 | 1 | 100% | 1 | 0.84 |
| bunch_530_jpg.rf.9fce365bbdacfaedee242b0 | 1 | 5 | 1 | 100% | 4 | 1.03 |
| bunch_544_jpg.rf.77d9672ecbc890fc8b14460 | 1 | 4 | 1 | 100% | 3 | 0.67 |
| bunch_555_jpg.rf.5c1742c197e80fed68431e9 | 1 | 3 | 1 | 100% | 2 | 0.67 |
| bunch_559_jpg.rf.c4ba31e84dedf364ee7441d | 1 | 6 | 1 | 100% | 5 | 0.65 |
| bunch_561_jpg.rf.3c66b5b9905e4bc2d221c53 | 1 | 4 | 1 | 100% | 3 | 0.61 |
| bunch_562_jpg.rf.0305d361f069c58873086bb | 1 | 1 | 1 | 100% | 0 | 0.43 |
| bunch_570_jpg.rf.39623afff4f3c3861b3f4b8 | 1 | 4 | 1 | 100% | 3 | 0.58 |
| bunch_575_jpg.rf.cb0225d1f2d39866d773330 | 1 | 7 | 1 | 100% | 6 | 0.67 |
| bunch_579_jpg.rf.5407cbad3dfc41244dd7ec6 | 1 | 2 | 1 | 100% | 1 | 0.45 |
| bunch_593_jpg.rf.facfe498c83392073b84c56 | 1 | 6 | 1 | 100% | 5 | 0.74 |
| bunch_611_jpg.rf.131d60f40ec7f35650e50d8 | 1 | 7 | 1 | 100% | 6 | 1.6 |
| bunch_78_jpg.rf.555d926aa73ea27af6c37eb7 | 1 | 7 | 1 | 100% | 6 | 0.93 |
| bunch_79_jpg.rf.4e14a318d3f4aea383f12b52 | 2 | 4 | 2 | 100% | 2 | 0.73 |
| bunch_82_jpg.rf.dea869305eddf28a1e4552c2 | 2 | 4 | 2 | 100% | 2 | 1.17 |
| bunch_92_jpg.rf.a004bc12cf3c0610cbd6e9a5 | 2 | 7 | 2 | 100% | 5 | 0.81 |
| bunch_97_jpg.rf.0ee9408bb835d780c8faff63 | 6 | 8 | 2 | 33% | 6 | 1.38 |
