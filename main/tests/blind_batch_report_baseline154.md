# Blind-batch report — live pipeline vs. Training labels

Batch size: 154 (limit 154)  ·  splits: valid + test  ·  detection: adaptive auto mode (conf 0.1, floor off)  ·  match IoU ≥ 0.3  ·  generated 2026-10-28 12:28

## Aggregates

| metric | value |
|---|---|
| images | 154 ok / 0 errored |
| mean detection recall (images with labels) | 68% over 152 |
| images with zero detections | 0 |
| matched bananas | 231 |
| ripeness agreement | 47 / 231 (20%) |
| spurious detections (no label overlap) | 615 |

## Confusion (label row × predicted col)

| label \ pred | overripe | ripe | semi-ripe | underripe | unlabelled |
|---|---|---|---|---|---|
| overripe | 7 | 0 | 0 | 0 | 0 |
| ripe | 89 | 2 | 0 | 1 | 0 |
| underripe | 75 | 19 | 12 | 26 | 0 |
| unlabelled | 0 | 0 | 0 | 0 | 0 |

## Mismatches (matched banana whose class disagrees)

| image | # | label | pred | G% | Y% | B% | det conf |
|---|---|---|---|---|---|---|---|
| bunch_647_jpg.rf.e9067b006cf5c | 3 | ripe | overripe | 0.0 | 52.6 | 41.8 | 0.15 |
| bunch_137_jpg.rf.afdb40dea5196 | 1 | ripe | overripe | 0.0 | 0.0 | 64.3 | 0.19 |
| bunch_147_jpg.rf.ab64917eab7cf | 1 | ripe | overripe | 0.0 | 60.5 | 96.3 | 0.41 |
| bunch_147_jpg.rf.ab64917eab7cf | 2 | ripe | overripe | 2.0 | 43.7 | 64.3 | 0.28 |
| bunch_232_jpg.rf.56806cb7233e3 | 3 | underripe | overripe | 0.1 | 86.0 | 29.9 | 0.33 |
| bunch_232_jpg.rf.56806cb7233e3 | 2 | ripe | overripe | 0.0 | 87.9 | 27.9 | 0.29 |
| bunch_232_jpg.rf.56806cb7233e3 | 1 | ripe | overripe | 0.3 | 99.4 | 42.6 | 0.66 |
| bunch_23_jpg.rf.62b7b717609d9b | 1 | ripe | overripe | 0.0 | 0.0 | 11.0 | 0.78 |
| bunch_289_jpg.rf.ebba34640b3e5 | 1 | ripe | overripe | 0.5 | 61.4 | 28.0 | 0.31 |
| bunch_290_jpg.rf.5bab7360124fe | 2 | ripe | overripe | 3.1 | 52.3 | 37.6 | 0.20 |
| bunch_290_jpg.rf.5bab7360124fe | 4 | ripe | overripe | 18.5 | 32.6 | 28.6 | 0.37 |
| bunch_292_jpg.rf.03ff4bdea42c1 | 1 | ripe | overripe | 0.2 | 67.8 | 80.6 | 0.34 |
| bunch_302_jpg.rf.c9b5780d91494 | 1 | ripe | overripe | 0.1 | 80.7 | 61.3 | 0.16 |
| bunch_302_jpg.rf.c9b5780d91494 | 3 | ripe | overripe | 0.1 | 74.0 | 75.5 | 0.13 |
| bunch_302_jpg.rf.c9b5780d91494 | 4 | ripe | overripe | 0.0 | 78.5 | 84.6 | 0.12 |
| bunch_317_jpg.rf.f0bf749b2019f | 2 | ripe | overripe | 0.0 | 70.4 | 72.2 | 0.66 |
| bunch_317_jpg.rf.f0bf749b2019f | 1 | ripe | overripe | 0.1 | 49.6 | 43.6 | 0.24 |
| bunch_327_jpg.rf.d8be7153a9ef3 | 2 | underripe | overripe | 3.9 | 61.9 | 43.1 | 0.21 |
| bunch_327_jpg.rf.d8be7153a9ef3 | 1 | ripe | overripe | 0.0 | 0.0 | 15.8 | 0.69 |
| bunch_329_jpg.rf.630135c02b4c5 | 3 | ripe | overripe | 0.7 | 51.4 | 51.8 | 0.27 |
| bunch_329_jpg.rf.630135c02b4c5 | 1 | ripe | overripe | 0.8 | 45.2 | 39.9 | 0.23 |
| bunch_329_jpg.rf.630135c02b4c5 | 8 | ripe | overripe | 0.0 | 38.3 | 23.8 | 0.18 |
| bunch_329_jpg.rf.630135c02b4c5 | 7 | ripe | overripe | 0.7 | 76.5 | 50.7 | 0.19 |
| bunch_329_jpg.rf.630135c02b4c5 | 4 | ripe | overripe | 0.0 | 99.2 | 25.8 | 0.46 |
| bunch_329_jpg.rf.630135c02b4c5 | 2 | ripe | overripe | 0.0 | 33.4 | 34.0 | 0.25 |
| bunch_329_jpg.rf.630135c02b4c5 | 5 | ripe | overripe | 0.0 | 99.0 | 43.6 | 0.25 |
| bunch_372_jpg.rf.1c13957182bd1 | 1 | ripe | overripe | 0.0 | 15.4 | 59.6 | 0.17 |
| bunch_374_jpg.rf.8a293009a0a10 | 1 | ripe | overripe | 0.0 | 2.3 | 84.5 | 0.30 |
| bunch_419_jpg.rf.d7ab8908bed8a | 1 | ripe | overripe | 0.0 | 80.6 | 89.9 | 0.20 |
| bunch_423_jpg.rf.1908ae9c3bf01 | 1 | ripe | overripe | 0.0 | 65.0 | 77.9 | 0.17 |
| bunch_427_jpg.rf.8aa3be5427be0 | 1 | ripe | overripe | 0.0 | 43.9 | 89.2 | 0.25 |
| bunch_432_jpg.rf.5a0099d014b03 | 1 | ripe | overripe | 0.0 | 58.7 | 81.9 | 0.39 |
| bunch_617_jpg.rf.306612ef2ad9e | 1 | ripe | overripe | 0.0 | 66.4 | 83.2 | 0.38 |
| bunch_80_jpg.rf.ca576b9f9efb0a | 3 | underripe | overripe | 4.1 | 99.0 | 39.2 | 0.22 |
| bunch_80_jpg.rf.ca576b9f9efb0a | 8 | underripe | overripe | 21.1 | 48.7 | 25.4 | 0.18 |
| bunch_80_jpg.rf.ca576b9f9efb0a | 1 | underripe | overripe | 0.0 | 0.0 | 6.3 | 0.53 |
| bunch_125_jpg.rf.4671325a15d80 | 2 | underripe | overripe | 0.7 | 91.6 | 58.5 | 0.30 |
| bunch_134_jpg.rf.7d66b62763ff7 | 1 | ripe | overripe | 4.3 | 27.4 | 69.2 | 0.31 |
| bunch_175_jpg.rf.4f8a43b7e0438 | 4 | underripe | overripe | 0.2 | 99.0 | 74.2 | 0.11 |
| bunch_181_jpg.rf.be7e5040b5e06 | 2 | underripe | ripe | 7.1 | 98.5 | 14.2 | 0.71 |
| bunch_181_jpg.rf.be7e5040b5e06 | 1 | underripe | overripe | 0.0 | 12.0 | 99.1 | 0.38 |
| bunch_181_jpg.rf.be7e5040b5e06 | 3 | underripe | overripe | 0.0 | 0.0 | 12.0 | 0.43 |
| bunch_184_jpg.rf.e6f1cc8448fbc | 2 | underripe | overripe | 3.7 | 75.2 | 24.5 | 0.35 |
| bunch_184_jpg.rf.e6f1cc8448fbc | 1 | ripe | overripe | 1.8 | 54.0 | 46.4 | 0.41 |
| bunch_187_jpg.rf.847fa6f325b41 | 1 | ripe | overripe | 0.0 | 55.4 | 96.8 | 0.38 |
| bunch_200_jpg.rf.68d01446a8d6e | 2 | underripe | overripe | 1.9 | 80.8 | 56.7 | 0.73 |
| bunch_200_jpg.rf.68d01446a8d6e | 1 | underripe | overripe | 3.7 | 66.0 | 38.8 | 0.28 |
| bunch_202_jpg.rf.bbcd851d25c8d | 2 | ripe | overripe | 0.0 | 71.2 | 78.8 | 0.27 |
| bunch_202_jpg.rf.bbcd851d25c8d | 1 | underripe | overripe | 0.0 | 78.5 | 82.4 | 0.41 |
| bunch_205_jpg.rf.330f12f1dff03 | 1 | underripe | overripe | 0.4 | 99.1 | 50.7 | 0.28 |
| bunch_208_jpg.rf.5a86923127bb7 | 1 | underripe | overripe | 0.6 | 80.3 | 57.3 | 0.22 |
| bunch_210_jpg.rf.d03f44cb4087d | 1 | underripe | overripe | 7.0 | 82.3 | 22.3 | 0.72 |
| bunch_210_jpg.rf.d03f44cb4087d | 2 | underripe | overripe | 0.0 | 0.0 | 14.0 | 0.38 |
| bunch_214_jpg.rf.24ae57a0f2e7f | 1 | ripe | overripe | 2.2 | 55.6 | 53.4 | 0.61 |
| bunch_224_jpg.rf.e3c35a3895bda | 3 | underripe | overripe | 0.0 | 64.2 | 98.5 | 0.42 |
| bunch_224_jpg.rf.e3c35a3895bda | 1 | ripe | overripe | 0.0 | 93.4 | 74.5 | 0.24 |
| bunch_225_jpg.rf.43f1993705592 | 1 | underripe | overripe | 0.0 | 96.9 | 38.0 | 0.54 |
| bunch_225_jpg.rf.43f1993705592 | 2 | underripe | overripe | 0.1 | 99.5 | 42.0 | 0.23 |
| bunch_227_jpg.rf.cf4c628ca7490 | 1 | underripe | overripe | 1.7 | 99.4 | 50.0 | 0.28 |
| bunch_252_jpg.rf.e5630742c2ae9 | 4 | underripe | ripe | 8.7 | 88.2 | 10.2 | 0.19 |
| bunch_252_jpg.rf.e5630742c2ae9 | 2 | underripe | ripe | 6.6 | 85.9 | 15.7 | 0.24 |
| bunch_252_jpg.rf.e5630742c2ae9 | 8 | underripe | overripe | 3.9 | 38.4 | 22.7 | 0.22 |
| bunch_253_jpg.rf.a9aa2b9ac03c5 | 4 | underripe | ripe | 5.3 | 84.6 | 12.2 | 0.91 |
| bunch_253_jpg.rf.a9aa2b9ac03c5 | 6 | underripe | ripe | 6.5 | 98.2 | 9.7 | 0.45 |
| bunch_273_jpg.rf.80fb5d16fee07 | 1 | underripe | overripe | 2.5 | 98.9 | 28.1 | 0.57 |
| bunch_295_jpg.rf.2c2127a1284c7 | 1 | ripe | overripe | 0.0 | 78.2 | 98.1 | 0.46 |
| bunch_295_jpg.rf.2c2127a1284c7 | 2 | underripe | overripe | 1.2 | 42.4 | 54.7 | 0.17 |
| bunch_307_jpg.rf.f896be506218c | 2 | underripe | overripe | 1.9 | 92.6 | 47.5 | 0.45 |
| bunch_392_jpg.rf.fe5acd1a3ea05 | 4 | underripe | ripe | 18.3 | 70.9 | 14.5 | 0.37 |
| bunch_392_jpg.rf.fe5acd1a3ea05 | 1 | underripe | overripe | 0.0 | 0.0 | 4.6 | 0.77 |
| bunch_444_jpg.rf.c77d1c225eca9 | 1 | underripe | overripe | 0.4 | 73.4 | 84.6 | 0.56 |
| bunch_445_jpg.rf.c22643daff6c9 | 1 | underripe | overripe | 0.0 | 78.1 | 91.7 | 0.81 |
| bunch_451_jpg.rf.46ddcb8ce13e7 | 1 | underripe | overripe | 1.2 | 79.2 | 74.8 | 0.25 |
| bunch_454_jpg.rf.d9602b659c8c7 | 1 | underripe | overripe | 0.8 | 74.2 | 71.6 | 0.43 |
| bunch_459_jpg.rf.e49b8d92e05b1 | 1 | underripe | overripe | 55.2 | 30.3 | 24.7 | 0.38 |
| bunch_459_jpg.rf.e49b8d92e05b1 | 3 | underripe | overripe | 11.7 | 98.3 | 31.9 | 0.28 |
| bunch_45_jpg.rf.ded71a67a7b9ff | 1 | underripe | overripe | 3.9 | 99.0 | 24.2 | 0.61 |
| bunch_462_jpg.rf.49bcee60c02ad | 1 | underripe | overripe | 0.0 | 38.8 | 98.9 | 0.49 |
| bunch_466_jpg.rf.d6722658d4152 | 1 | ripe | overripe | 0.0 | 37.2 | 98.6 | 0.13 |
| bunch_46_jpg.rf.c67a13a7a11f47 | 3 | underripe | overripe | 0.1 | 40.4 | 33.5 | 0.22 |
| bunch_508_jpg.rf.1bf58f1f85493 | 1 | underripe | overripe | 8.3 | 98.3 | 32.6 | 0.59 |
| bunch_509_jpg.rf.17751229d96d1 | 1 | underripe | overripe | 2.8 | 98.8 | 48.0 | 0.22 |
| bunch_556_jpg.rf.060134afaaff5 | 1 | underripe | overripe | 0.0 | 0.0 | 52.6 | 0.74 |
| bunch_571_jpg.rf.5f738bda47ba1 | 1 | underripe | ripe | 7.8 | 99.5 | 9.8 | 0.88 |
| bunch_598_jpg.rf.13cfe98a228f6 | 1 | underripe | overripe | 5.2 | 98.3 | 26.6 | 0.68 |
| bunch_608_jpg.rf.7490bd7bb1932 | 1 | underripe | overripe | 50.2 | 26.5 | 25.8 | 0.36 |
| bunch_65_jpg.rf.3797af5fe0ed5f | 3 | underripe | ripe | 15.7 | 87.0 | 2.5 | 0.36 |
| bunch_65_jpg.rf.3797af5fe0ed5f | 2 | underripe | ripe | 12.0 | 98.3 | 17.5 | 0.32 |
| bunch_101_jpg.rf.d1a1e7823b929 | 1 | ripe | overripe | 0.0 | 29.5 | 99.0 | 0.25 |
| bunch_111_jpg.rf.51b36f6d39aaa | 3 | ripe | overripe | 0.0 | 67.5 | 98.7 | 0.20 |
| bunch_111_jpg.rf.51b36f6d39aaa | 2 | ripe | overripe | 0.0 | 63.9 | 95.4 | 0.32 |
| bunch_116_jpg.rf.9f4119431a28d | 2 | ripe | overripe | 0.5 | 45.9 | 55.8 | 0.39 |
| bunch_123_jpg.rf.ca8772a96f82a | 2 | ripe | overripe | 0.2 | 62.7 | 89.3 | 0.79 |
| bunch_123_jpg.rf.ca8772a96f82a | 1 | ripe | overripe | 0.0 | 37.4 | 96.1 | 0.30 |
| bunch_127_jpg.rf.af196dbf72d76 | 2 | ripe | overripe | 0.0 | 46.7 | 69.7 | 0.64 |
| bunch_127_jpg.rf.af196dbf72d76 | 4 | ripe | overripe | 0.0 | 24.0 | 99.5 | 0.35 |
| bunch_14_jpg.rf.e403f859d62c37 | 2 | ripe | overripe | 0.1 | 87.8 | 45.0 | 0.71 |
| bunch_183_jpg.rf.27b9110a9906e | 2 | ripe | overripe | 3.5 | 74.1 | 52.8 | 0.52 |
| bunch_185_jpg.rf.b621f38f16a29 | 2 | ripe | overripe | 0.0 | 62.5 | 76.6 | 0.34 |
| bunch_185_jpg.rf.b621f38f16a29 | 1 | underripe | overripe | 0.0 | 98.8 | 50.5 | 0.35 |
| bunch_24_jpg.rf.69a15577015a4c | 1 | ripe | overripe | 3.6 | 35.3 | 45.9 | 0.27 |
| bunch_26_jpg.rf.519721a7f222b7 | 1 | ripe | overripe | 0.0 | 11.4 | 96.2 | 0.41 |
| bunch_277_jpg.rf.3d0db1099ae4b | 1 | ripe | overripe | 0.8 | 99.3 | 56.3 | 0.37 |
| bunch_278_jpg.rf.04f2d116273c5 | 1 | ripe | overripe | 0.4 | 74.9 | 66.9 | 0.24 |
| bunch_27_jpg.rf.7b66ec71ac176f | 3 | ripe | overripe | 0.5 | 46.0 | 88.3 | 0.18 |
| bunch_27_jpg.rf.7b66ec71ac176f | 1 | ripe | overripe | 0.5 | 32.8 | 64.8 | 0.16 |
| bunch_280_jpg.rf.81f558727fc9e | 1 | ripe | overripe | 4.6 | 72.8 | 38.2 | 0.45 |
| bunch_294_jpg.rf.80e68e3ed0ad2 | 1 | ripe | overripe | 0.0 | 83.6 | 89.9 | 0.85 |
| bunch_340_jpg.rf.1cf347b36f4b7 | 1 | ripe | overripe | 0.0 | 71.3 | 75.0 | 0.14 |
| bunch_340_jpg.rf.1cf347b36f4b7 | 2 | ripe | overripe | 0.0 | 61.4 | 69.1 | 0.41 |
| bunch_348_jpg.rf.160173c7f65ef | 1 | ripe | overripe | 0.0 | 65.3 | 55.9 | 0.20 |
| bunch_348_jpg.rf.160173c7f65ef | 4 | ripe | overripe | 0.0 | 80.7 | 69.3 | 0.46 |
| bunch_348_jpg.rf.160173c7f65ef | 6 | ripe | overripe | 1.0 | 91.4 | 48.1 | 0.27 |
| bunch_360_jpg.rf.652a6a7ef71e7 | 1 | ripe | overripe | 0.0 | 61.8 | 90.1 | 0.24 |
| bunch_377_jpg.rf.9b55bd2448ef3 | 3 | ripe | overripe | 0.0 | 22.5 | 98.1 | 0.32 |
| bunch_385_jpg.rf.328b84f40d4ee | 1 | ripe | overripe | 0.7 | 82.3 | 74.6 | 0.66 |
| bunch_38_jpg.rf.26d0f72f51d3b1 | 3 | ripe | overripe | 0.1 | 32.0 | 52.0 | 0.27 |
| bunch_38_jpg.rf.26d0f72f51d3b1 | 2 | ripe | overripe | 8.1 | 36.5 | 69.2 | 0.29 |
| bunch_401_jpg.rf.08b91df5d734f | 2 | ripe | overripe | 0.0 | 42.9 | 54.2 | 0.42 |
| bunch_401_jpg.rf.08b91df5d734f | 7 | ripe | overripe | 0.0 | 98.7 | 53.2 | 0.47 |
| bunch_417_jpg.rf.cf83da72e6538 | 1 | ripe | overripe | 0.0 | 78.1 | 61.1 | 0.45 |
| bunch_465_jpg.rf.b4d8dca50180a | 1 | ripe | overripe | 1.3 | 70.3 | 44.7 | 0.16 |
| bunch_465_jpg.rf.b4d8dca50180a | 3 | ripe | overripe | 1.4 | 98.3 | 82.1 | 0.20 |
| bunch_465_jpg.rf.b4d8dca50180a | 5 | underripe | overripe | 0.0 | 68.1 | 92.8 | 0.14 |
| bunch_465_jpg.rf.b4d8dca50180a | 2 | ripe | overripe | 2.0 | 65.3 | 63.1 | 0.32 |
| bunch_471_jpg.rf.24bb890b1d5e8 | 5 | ripe | overripe | 0.1 | 45.5 | 94.6 | 0.28 |
| bunch_47_jpg.rf.e2f2b5d445a28d | 1 | ripe | overripe | 0.1 | 84.9 | 33.6 | 0.23 |
| bunch_47_jpg.rf.e2f2b5d445a28d | 2 | ripe | overripe | 0.0 | 55.0 | 97.1 | 0.54 |
| bunch_47_jpg.rf.e2f2b5d445a28d | 3 | underripe | ripe | 7.7 | 98.7 | 12.9 | 0.31 |
| bunch_68_jpg.rf.f3b075eeb58160 | 3 | ripe | overripe | 2.9 | 88.0 | 73.9 | 0.46 |
| bunch_68_jpg.rf.f3b075eeb58160 | 4 | ripe | overripe | 0.1 | 99.5 | 36.4 | 0.29 |
| bunch_69_jpg.rf.3ead8efa0a292a | 3 | ripe | overripe | 0.2 | 78.7 | 72.4 | 0.25 |
| bunch_69_jpg.rf.3ead8efa0a292a | 1 | ripe | overripe | 0.0 | 97.1 | 45.0 | 0.27 |
| bunch_69_jpg.rf.3ead8efa0a292a | 2 | ripe | overripe | 0.1 | 89.7 | 54.8 | 0.54 |
| bunch_112_jpg.rf.6b61510258457 | 2 | ripe | overripe | 0.0 | 81.0 | 91.8 | 0.21 |
| bunch_112_jpg.rf.6b61510258457 | 4 | ripe | overripe | 0.0 | 0.0 | 2.3 | 0.37 |
| bunch_171_jpg.rf.7d55d6339b026 | 5 | underripe | overripe | 0.4 | 80.0 | 66.1 | 0.12 |
| bunch_171_jpg.rf.7d55d6339b026 | 4 | underripe | overripe | 0.0 | 78.7 | 63.9 | 0.19 |
| bunch_171_jpg.rf.7d55d6339b026 | 3 | underripe | ripe | 1.8 | 99.6 | 14.7 | 0.19 |
| bunch_197_jpg.rf.997be390a5c7b | 1 | underripe | overripe | 2.2 | 84.8 | 31.4 | 0.67 |
| bunch_236_jpg.rf.df498500f60b3 | 1 | underripe | overripe | 0.0 | 88.2 | 75.2 | 0.36 |
| bunch_236_jpg.rf.df498500f60b3 | 2 | underripe | overripe | 0.0 | 80.9 | 72.0 | 0.13 |
| bunch_241_jpg.rf.f8fe539bb1abf | 1 | underripe | overripe | 0.4 | 60.3 | 65.8 | 0.50 |
| bunch_241_jpg.rf.f8fe539bb1abf | 3 | underripe | overripe | 0.1 | 99.2 | 78.3 | 0.72 |
| bunch_246_jpg.rf.937d8a75eaba0 | 1 | underripe | overripe | 1.9 | 36.3 | 54.4 | 0.45 |
| bunch_256_jpg.rf.c5eef63f143a9 | 4 | underripe | overripe | 6.1 | 44.7 | 50.3 | 0.15 |
| bunch_256_jpg.rf.c5eef63f143a9 | 3 | underripe | ripe | 13.1 | 97.4 | 3.7 | 0.16 |
| bunch_260_jpg.rf.670bb5f5ba073 | 1 | ripe | overripe | 8.0 | 38.9 | 23.0 | 0.83 |
| bunch_260_jpg.rf.670bb5f5ba073 | 2 | underripe | ripe | 9.7 | 40.9 | 15.3 | 0.57 |
| bunch_267_jpg.rf.1eaefd3d149e5 | 1 | underripe | overripe | 2.7 | 89.0 | 50.4 | 0.56 |
| bunch_267_jpg.rf.1eaefd3d149e5 | 2 | ripe | unripe | 31.2 | 24.2 | 4.2 | 0.26 |
| bunch_291_jpg.rf.e5dca3811156e | 2 | underripe | overripe | 1.2 | 98.9 | 62.2 | 0.19 |
| bunch_291_jpg.rf.e5dca3811156e | 1 | ripe | overripe | 0.0 | 0.0 | 8.3 | 0.47 |
| bunch_304_jpg.rf.a3dd6998a5c6d | 1 | underripe | overripe | 1.0 | 50.0 | 70.7 | 0.15 |
| bunch_304_jpg.rf.a3dd6998a5c6d | 3 | underripe | overripe | 2.5 | 97.7 | 31.8 | 0.38 |
| bunch_309_jpg.rf.634a3be8a108d | 4 | underripe | ripe | 19.0 | 66.7 | 10.7 | 0.18 |
| bunch_309_jpg.rf.634a3be8a108d | 2 | underripe | overripe | 0.0 | 0.0 | 1.9 | 0.46 |
| bunch_350_jpg.rf.030c1c50dc2ee | 2 | underripe | overripe | 0.4 | 83.0 | 58.3 | 0.50 |
| bunch_350_jpg.rf.030c1c50dc2ee | 4 | ripe | overripe | 1.9 | 56.1 | 41.3 | 0.71 |
| bunch_359_jpg.rf.04c266ec33ac9 | 2 | underripe | ripe | 13.3 | 98.0 | 14.0 | 0.56 |
| bunch_359_jpg.rf.04c266ec33ac9 | 5 | underripe | overripe | 2.1 | 66.7 | 49.9 | 0.32 |
| bunch_359_jpg.rf.04c266ec33ac9 | 1 | underripe | overripe | 1.0 | 63.3 | 69.7 | 0.35 |
| bunch_42_jpg.rf.517a873a24fbb0 | 6 | underripe | ripe | 16.6 | 55.9 | 5.1 | 0.66 |
| bunch_44_jpg.rf.fa217c553450b1 | 1 | underripe | overripe | 32.4 | 34.2 | 21.0 | 0.66 |
| bunch_452_jpg.rf.c7855cb95a26c | 1 | underripe | overripe | 0.0 | 96.3 | 70.7 | 0.31 |
| bunch_474_jpg.rf.63445b0e61e23 | 1 | underripe | overripe | 0.6 | 99.0 | 46.1 | 0.66 |
| bunch_475_jpg.rf.ea005c7f50ced | 1 | underripe | overripe | 1.7 | 67.4 | 56.1 | 0.14 |
| bunch_482_jpg.rf.032a1fe534c97 | 1 | underripe | overripe | 0.4 | 99.1 | 44.8 | 0.80 |
| bunch_48_jpg.rf.e0ea423ca4aa11 | 1 | underripe | ripe | 2.4 | 99.4 | 12.9 | 0.46 |
| bunch_50_jpg.rf.d773a3e466f02c | 1 | underripe | overripe | 11.9 | 70.3 | 48.7 | 0.30 |
| bunch_559_jpg.rf.c4ba31e84dedf | 1 | underripe | ripe | 15.1 | 81.7 | 18.3 | 0.39 |
| bunch_575_jpg.rf.cb0225d1f2d39 | 1 | underripe | overripe | 5.3 | 62.0 | 56.8 | 0.28 |
| bunch_579_jpg.rf.5407cbad3dfc4 | 1 | underripe | overripe | 41.3 | 27.1 | 27.1 | 0.13 |
| bunch_593_jpg.rf.facfe498c8339 | 1 | underripe | overripe | 3.1 | 78.7 | 61.1 | 0.34 |
| bunch_611_jpg.rf.131d60f40ec7f | 1 | underripe | overripe | 0.0 | 59.2 | 97.7 | 0.14 |
| bunch_78_jpg.rf.555d926aa73ea2 | 1 | underripe | overripe | 0.6 | 98.6 | 27.4 | 0.63 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 1 | underripe | overripe | 0.2 | 78.8 | 50.6 | 0.56 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 2 | underripe | ripe | 3.3 | 99.6 | 16.9 | 0.76 |
| bunch_82_jpg.rf.dea869305eddf2 | 2 | ripe | overripe | 0.3 | 31.5 | 57.4 | 0.74 |
| bunch_82_jpg.rf.dea869305eddf2 | 1 | underripe | overripe | 2.4 | 50.9 | 60.7 | 0.44 |
| bunch_92_jpg.rf.a004bc12cf3c06 | 1 | underripe | overripe | 1.3 | 11.8 | 18.8 | 0.23 |
| bunch_92_jpg.rf.a004bc12cf3c06 | 2 | ripe | overripe | 0.0 | 40.9 | 99.1 | 0.19 |
| bunch_97_jpg.rf.0ee9408bb835d7 | 1 | underripe | overripe | 1.7 | 99.0 | 21.8 | 0.28 |
| bunch_97_jpg.rf.0ee9408bb835d7 | 4 | underripe | overripe | 2.2 | 33.3 | 35.8 | 0.57 |

## Per-image detail

| image | labels | dets | matched | recall | spurious | sec |
|---|---|---|---|---|---|---|
| bunch_100_jpg.rf.a1b2603b38385cde1eb1782 | 1 | 2 | 0 | 0% | 2 | 5.14 |
| bunch_643_jpg.rf.67f35b648ec4dfd74dbfae1 | 6 | 3 | 2 | 33% | 1 | 0.93 |
| bunch_647_jpg.rf.e9067b006cf5ca75b95818c | 4 | 10 | 3 | 75% | 7 | 1.35 |
| bunch_120_jpg.rf.5cc9f4e5b3eca6edbd3611a | 2 | 6 | 0 | 0% | 6 | 0.61 |
| bunch_137_jpg.rf.afdb40dea51966e920b1427 | 7 | 9 | 1 | 14% | 8 | 0.75 |
| bunch_147_jpg.rf.ab64917eab7cfa2db5b3852 | 3 | 17 | 2 | 67% | 15 | 1.15 |
| bunch_232_jpg.rf.56806cb7233e30f1c313d6a | 3 | 7 | 3 | 100% | 4 | 0.51 |
| bunch_23_jpg.rf.62b7b717609d9bae9a083523 | 1 | 1 | 1 | 100% | 0 | 0.64 |
| bunch_289_jpg.rf.ebba34640b3e54eacc6afa0 | 3 | 4 | 1 | 33% | 3 | 0.36 |
| bunch_290_jpg.rf.5bab7360124fe3fde2cd6ed | 4 | 10 | 3 | 75% | 7 | 0.78 |
| bunch_292_jpg.rf.03ff4bdea42c1e84dcbc902 | 2 | 8 | 2 | 100% | 6 | 0.93 |
| bunch_302_jpg.rf.c9b5780d914946c6e79f1f9 | 8 | 7 | 3 | 38% | 4 | 0.98 |
| bunch_317_jpg.rf.f0bf749b2019f1d4447fa19 | 5 | 5 | 2 | 40% | 3 | 1.07 |
| bunch_327_jpg.rf.d8be7153a9ef3a49a86c3fa | 3 | 3 | 2 | 67% | 1 | 0.61 |
| bunch_329_jpg.rf.630135c02b4c569025e27ba | 8 | 26 | 7 | 88% | 19 | 0.88 |
| bunch_372_jpg.rf.1c13957182bd1fb9f774953 | 1 | 6 | 1 | 100% | 5 | 0.79 |
| bunch_374_jpg.rf.8a293009a0a102e6d504ee8 | 2 | 4 | 1 | 50% | 3 | 0.55 |
| bunch_419_jpg.rf.d7ab8908bed8a7954446f21 | 1 | 4 | 1 | 100% | 3 | 0.63 |
| bunch_423_jpg.rf.1908ae9c3bf01ef4fd8a6a5 | 1 | 4 | 1 | 100% | 3 | 1.11 |
| bunch_427_jpg.rf.8aa3be5427be0c48314f836 | 1 | 4 | 1 | 100% | 3 | 0.69 |
| bunch_430_jpg.rf.2ed80d4fbecdb13ff3af2d0 | 1 | 2 | 0 | 0% | 2 | 0.4 |
| bunch_432_jpg.rf.5a0099d014b0321991e3188 | 1 | 8 | 1 | 100% | 7 | 1.79 |
| bunch_617_jpg.rf.306612ef2ad9e6208969938 | 1 | 7 | 1 | 100% | 6 | 0.82 |
| bunch_80_jpg.rf.ca576b9f9efb0a1438b80c64 | 8 | 4 | 3 | 38% | 1 | 0.84 |
| bunch_87_jpg.rf.33cb34b7da304b1a98f3905b | 1 | 3 | 0 | 0% | 3 | 0.55 |
| bunch_125_jpg.rf.4671325a15d8078b7f5ca10 | 2 | 5 | 1 | 50% | 4 | 0.33 |
| bunch_134_jpg.rf.7d66b62763ff7e210845470 | 6 | 9 | 1 | 17% | 8 | 0.86 |
| bunch_165_jpg.rf.73392960260d47ff5fc58a4 | 3 | 2 | 1 | 33% | 1 | 0.79 |
| bunch_175_jpg.rf.4f8a43b7e0438bbdfcf338f | 4 | 5 | 2 | 50% | 3 | 0.52 |
| bunch_181_jpg.rf.be7e5040b5e06142d6a1120 | 3 | 3 | 3 | 100% | 0 | 0.71 |
| bunch_184_jpg.rf.e6f1cc8448fbc5de2213b48 | 3 | 5 | 2 | 67% | 3 | 0.69 |
| bunch_187_jpg.rf.847fa6f325b41affd82f049 | 2 | 7 | 2 | 100% | 5 | 0.94 |
| bunch_200_jpg.rf.68d01446a8d6ef7180b7541 | 2 | 8 | 2 | 100% | 6 | 0.67 |
| bunch_202_jpg.rf.bbcd851d25c8d78fed28b3f | 2 | 5 | 2 | 100% | 3 | 0.43 |
| bunch_205_jpg.rf.330f12f1dff03d0fcedcd0f | 4 | 2 | 1 | 25% | 1 | 0.65 |
| bunch_208_jpg.rf.5a86923127bb7ef96bd9141 | 2 | 1 | 1 | 50% | 0 | 0.28 |
| bunch_210_jpg.rf.d03f44cb4087d79f8cf5d64 | 3 | 2 | 2 | 67% | 0 | 0.31 |
| bunch_214_jpg.rf.24ae57a0f2e7fee73bdf2bc | 2 | 8 | 1 | 50% | 7 | 0.49 |
| bunch_224_jpg.rf.e3c35a3895bda9717a4bbb9 | 3 | 6 | 2 | 67% | 4 | 0.69 |
| bunch_225_jpg.rf.43f19937055926a72ba72af | 2 | 11 | 2 | 100% | 9 | 1.18 |
| bunch_227_jpg.rf.cf4c628ca749030e63b2cd3 | 1 | 9 | 1 | 100% | 8 | 0.75 |
| bunch_252_jpg.rf.e5630742c2ae9f155c9fb0e | 8 | 5 | 3 | 38% | 2 | 0.54 |
| bunch_253_jpg.rf.a9aa2b9ac03c5fcc5e8009a | 7 | 2 | 2 | 29% | 0 | 0.55 |
| bunch_273_jpg.rf.80fb5d16fee0753d30787ce | 1 | 5 | 1 | 100% | 4 | 0.76 |
| bunch_295_jpg.rf.2c2127a1284c71f64be4054 | 3 | 12 | 2 | 67% | 10 | 1.08 |
| bunch_307_jpg.rf.f896be506218c7f026cdd5e | 5 | 7 | 2 | 40% | 5 | 0.77 |
| bunch_313_jpg.rf.462dad60241ac00dcac6960 | 3 | 4 | 2 | 67% | 2 | 0.82 |
| bunch_387_jpg.rf.8693a9674f78ebb1415dc98 | 3 | 7 | 2 | 67% | 5 | 0.92 |
| bunch_392_jpg.rf.fe5acd1a3ea05e88f65b1ef | 4 | 3 | 2 | 50% | 1 | 0.8 |
| bunch_444_jpg.rf.c77d1c225eca9275b2ca413 | 1 | 5 | 1 | 100% | 4 | 0.68 |
| bunch_445_jpg.rf.c22643daff6c95bd477458b | 1 | 1 | 1 | 100% | 0 | 0.34 |
| bunch_451_jpg.rf.46ddcb8ce13e73d3bcb4b65 | 1 | 8 | 1 | 100% | 7 | 1.36 |
| bunch_454_jpg.rf.d9602b659c8c77bf5940134 | 1 | 7 | 1 | 100% | 6 | 1.05 |
| bunch_459_jpg.rf.e49b8d92e05b1e2b08ed2a7 | 4 | 7 | 2 | 50% | 5 | 0.75 |
| bunch_45_jpg.rf.ded71a67a7b9ff84df88a891 | 4 | 7 | 3 | 75% | 4 | 0.27 |
| bunch_462_jpg.rf.49bcee60c02ad4156a6a7ba | 5 | 9 | 1 | 20% | 8 | 1.78 |
| bunch_466_jpg.rf.d6722658d41529496258d47 | 5 | 2 | 1 | 20% | 1 | 1.03 |
| bunch_46_jpg.rf.c67a13a7a11f47070f32f45f | 3 | 2 | 1 | 33% | 1 | 0.24 |
| bunch_508_jpg.rf.1bf58f1f8549386aa36872f | 1 | 3 | 1 | 100% | 2 | 0.58 |
| bunch_509_jpg.rf.17751229d96d19bbcf1d55e | 1 | 5 | 1 | 100% | 4 | 1.06 |
| bunch_516_jpg.rf.49a31f9cd5a23f8f3de0315 | 1 | 1 | 1 | 100% | 0 | 0.58 |
| bunch_519_jpg.rf.dabeda794cb942c0866c8f1 | 1 | 2 | 1 | 100% | 1 | 1.0 |
| bunch_520_jpg.rf.6217bf94a1004735a44ade8 | 1 | 3 | 1 | 100% | 2 | 0.68 |
| bunch_521_jpg.rf.9eb6fab1cdd1c4b4f8c95bd | 1 | 3 | 1 | 100% | 2 | 1.1 |
| bunch_525_jpg.rf.d9a175de13db2946f74a0ea | 1 | 1 | 1 | 100% | 0 | 0.52 |
| bunch_556_jpg.rf.060134afaaff5e1a222901d | 1 | 2 | 1 | 100% | 1 | 0.79 |
| bunch_55_jpg.rf.02611d3d6408d695dd2d0bd6 | 3 | 1 | 1 | 33% | 0 | 0.25 |
| bunch_569_jpg.rf.90c1d2a7f72d6d8ed9c6848 | 1 | 1 | 1 | 100% | 0 | 0.67 |
| bunch_571_jpg.rf.5f738bda47ba13b5fb1c271 | 1 | 2 | 1 | 100% | 1 | 1.22 |
| bunch_596_jpg.rf.351ddaf6d73fda837e51fdd | 1 | 2 | 0 | 0% | 2 | 0.34 |
| bunch_598_jpg.rf.13cfe98a228f6ea56795f12 | 1 | 3 | 1 | 100% | 2 | 0.81 |
| bunch_608_jpg.rf.7490bd7bb193256bdc6f7df | 1 | 6 | 1 | 100% | 5 | 1.27 |
| bunch_633_jpeg.rf.c0d7e698897ba31c9d3209 | 4 | 2 | 2 | 50% | 0 | 0.22 |
| bunch_634_jpeg.rf.d65fd2ab2655ea500d6937 | 2 | 1 | 1 | 50% | 0 | 0.19 |
| bunch_65_jpg.rf.3797af5fe0ed5f1317ec0d93 | 3 | 9 | 3 | 100% | 6 | 0.73 |
| null_33_jpg.rf.96935df24891f67395b37f95d | 0 | 2 | 0 | — | 2 | 0.19 |
| null_49_jpg.rf.b484eec1c500cc7b4c8687492 | 0 | 9 | 0 | — | 9 | 0.65 |
| bunch_426_jpg.rf.63d4554df0df892a4a1b968 | 1 | 1 | 1 | 100% | 0 | 0.92 |
| bunch_440_jpg.rf.9ae90b20e93d48a5599b6aa | 1 | 1 | 1 | 100% | 0 | 0.3 |
| bunch_618_jpg.rf.3dc1311a5cf991afaa76c5d | 1 | 4 | 0 | 0% | 4 | 0.59 |
| bunch_84_jpg.rf.2a87ae6c1e86623b3ba1ca67 | 1 | 3 | 0 | 0% | 3 | 0.24 |
| bunch_101_jpg.rf.d1a1e7823b929082e9a9f4f | 3 | 1 | 1 | 33% | 0 | 0.74 |
| bunch_111_jpg.rf.51b36f6d39aaab77633ae01 | 3 | 6 | 2 | 67% | 4 | 1.09 |
| bunch_116_jpg.rf.9f4119431a28dd43a2fc8fa | 6 | 14 | 1 | 17% | 13 | 0.87 |
| bunch_123_jpg.rf.ca8772a96f82a4b736f550a | 2 | 5 | 2 | 100% | 3 | 1.03 |
| bunch_127_jpg.rf.af196dbf72d768b89914fa2 | 5 | 2 | 2 | 40% | 0 | 0.25 |
| bunch_14_jpg.rf.e403f859d62c37e09bc493f7 | 4 | 7 | 1 | 25% | 6 | 0.52 |
| bunch_154_jpg.rf.1a1e0ebdb6f1aff6584d0ec | 2 | 4 | 0 | 0% | 4 | 0.44 |
| bunch_183_jpg.rf.27b9110a9906eada0046429 | 3 | 3 | 1 | 33% | 2 | 0.38 |
| bunch_185_jpg.rf.b621f38f16a295aebbd47e2 | 3 | 9 | 2 | 67% | 7 | 0.68 |
| bunch_24_jpg.rf.69a15577015a4c892d889e6c | 3 | 9 | 1 | 33% | 8 | 1.32 |
| bunch_26_jpg.rf.519721a7f222b756fcc53eeb | 5 | 7 | 1 | 20% | 6 | 1.24 |
| bunch_277_jpg.rf.3d0db1099ae4b31f2a81d91 | 1 | 3 | 1 | 100% | 2 | 0.44 |
| bunch_278_jpg.rf.04f2d116273c5b4c19a0c2d | 2 | 4 | 1 | 50% | 3 | 0.74 |
| bunch_27_jpg.rf.7b66ec71ac176fc7a857b19b | 4 | 13 | 2 | 50% | 11 | 1.14 |
| bunch_280_jpg.rf.81f558727fc9e3416796e0b | 1 | 3 | 1 | 100% | 2 | 0.66 |
| bunch_294_jpg.rf.80e68e3ed0ad201a58005ed | 4 | 2 | 1 | 25% | 1 | 0.46 |
| bunch_340_jpg.rf.1cf347b36f4b7268be775aa | 5 | 10 | 2 | 40% | 8 | 0.91 |
| bunch_348_jpg.rf.160173c7f65efe746b4ad7a | 8 | 15 | 3 | 38% | 12 | 0.86 |
| bunch_360_jpg.rf.652a6a7ef71e74f52af3f58 | 4 | 10 | 1 | 25% | 9 | 0.53 |
| bunch_377_jpg.rf.9b55bd2448ef3f6ca814f5a | 3 | 1 | 1 | 33% | 0 | 0.66 |
| bunch_385_jpg.rf.328b84f40d4ee1f6780c1c6 | 1 | 5 | 1 | 100% | 4 | 0.96 |
| bunch_38_jpg.rf.26d0f72f51d3b1faaa1f1bcc | 6 | 14 | 3 | 50% | 11 | 1.43 |
| bunch_401_jpg.rf.08b91df5d734ffa78ab1237 | 9 | 4 | 2 | 22% | 2 | 0.63 |
| bunch_417_jpg.rf.cf83da72e653889888810b9 | 1 | 11 | 1 | 100% | 10 | 1.3 |
| bunch_465_jpg.rf.b4d8dca50180a5ccf8262c2 | 6 | 11 | 4 | 67% | 7 | 0.98 |
| bunch_471_jpg.rf.24bb890b1d5e8dcce658c95 | 7 | 7 | 1 | 14% | 6 | 1.11 |
| bunch_47_jpg.rf.e2f2b5d445a28dd710d4dbe5 | 3 | 5 | 3 | 100% | 2 | 0.3 |
| bunch_68_jpg.rf.f3b075eeb5816019c0fb035b | 5 | 12 | 2 | 40% | 10 | 0.86 |
| bunch_69_jpg.rf.3ead8efa0a292abb523b318a | 3 | 12 | 3 | 100% | 9 | 1.3 |
| bunch_112_jpg.rf.6b615102584573c3eb2958b | 4 | 9 | 2 | 50% | 7 | 0.88 |
| bunch_167_jpg.rf.fae2d400e1fe7eed040be78 | 6 | 8 | 3 | 50% | 5 | 0.56 |
| bunch_171_jpg.rf.7d55d6339b026093a56ed1b | 5 | 4 | 4 | 80% | 0 | 0.59 |
| bunch_197_jpg.rf.997be390a5c7bc59bcb96ae | 1 | 2 | 1 | 100% | 1 | 0.41 |
| bunch_236_jpg.rf.df498500f60b3cc0533a645 | 3 | 6 | 2 | 67% | 4 | 1.1 |
| bunch_241_jpg.rf.f8fe539bb1abff8b221e71c | 3 | 5 | 2 | 67% | 3 | 0.87 |
| bunch_246_jpg.rf.937d8a75eaba0bc8f250539 | 1 | 2 | 1 | 100% | 1 | 0.53 |
| bunch_256_jpg.rf.c5eef63f143a94c5af914dd | 4 | 12 | 2 | 50% | 10 | 0.82 |
| bunch_260_jpg.rf.670bb5f5ba073621b18fd12 | 2 | 4 | 2 | 100% | 2 | 0.66 |
| bunch_267_jpg.rf.1eaefd3d149e5c1f101d31e | 2 | 7 | 2 | 100% | 5 | 1.04 |
| bunch_291_jpg.rf.e5dca3811156e90426e85a1 | 2 | 11 | 2 | 100% | 9 | 0.95 |
| bunch_304_jpg.rf.a3dd6998a5c6d17776679aa | 4 | 12 | 3 | 75% | 9 | 1.02 |
| bunch_309_jpg.rf.634a3be8a108de97d127163 | 4 | 7 | 2 | 50% | 5 | 0.97 |
| bunch_350_jpg.rf.030c1c50dc2ee956cd145da | 4 | 5 | 2 | 50% | 3 | 0.93 |
| bunch_359_jpg.rf.04c266ec33ac97a51122ecc | 7 | 5 | 4 | 57% | 1 | 0.53 |
| bunch_395_jpg.rf.7244f7360becf46248e97c7 | 2 | 7 | 1 | 50% | 6 | 0.98 |
| bunch_396_jpg.rf.f710a7bc921353787a05f28 | 2 | 1 | 1 | 50% | 0 | 0.65 |
| bunch_398_jpg.rf.a64a59303d20f8a7935d9ad | 1 | 5 | 1 | 100% | 4 | 1.55 |
| bunch_42_jpg.rf.517a873a24fbb0d2c9d1298b | 7 | 1 | 1 | 14% | 0 | 0.28 |
| bunch_44_jpg.rf.fa217c553450b1450863a9ef | 1 | 1 | 1 | 100% | 0 | 0.18 |
| bunch_452_jpg.rf.c7855cb95a26cb93abc8b8c | 1 | 2 | 1 | 100% | 1 | 0.47 |
| bunch_474_jpg.rf.63445b0e61e2372be30c7d5 | 1 | 3 | 1 | 100% | 2 | 0.5 |
| bunch_475_jpg.rf.ea005c7f50ced29d8a3f67e | 1 | 8 | 1 | 100% | 7 | 0.77 |
| bunch_482_jpg.rf.032a1fe534c9763bd29f38b | 1 | 5 | 1 | 100% | 4 | 0.71 |
| bunch_48_jpg.rf.e0ea423ca4aa114124de2151 | 1 | 3 | 1 | 100% | 2 | 0.3 |
| bunch_49_jpg.rf.2ecd888d042b512fc58c28d9 | 2 | 1 | 1 | 50% | 0 | 0.22 |
| bunch_50_jpg.rf.d773a3e466f02c060b1d6c5c | 2 | 8 | 1 | 50% | 7 | 0.19 |
| bunch_517_jpg.rf.87bb8c04473353edd7f9ec4 | 1 | 2 | 1 | 100% | 1 | 0.79 |
| bunch_530_jpg.rf.9fce365bbdacfaedee242b0 | 1 | 5 | 1 | 100% | 4 | 0.94 |
| bunch_544_jpg.rf.77d9672ecbc890fc8b14460 | 1 | 4 | 1 | 100% | 3 | 0.53 |
| bunch_555_jpg.rf.5c1742c197e80fed68431e9 | 1 | 3 | 1 | 100% | 2 | 0.53 |
| bunch_559_jpg.rf.c4ba31e84dedf364ee7441d | 1 | 6 | 1 | 100% | 5 | 0.68 |
| bunch_561_jpg.rf.3c66b5b9905e4bc2d221c53 | 1 | 4 | 1 | 100% | 3 | 0.59 |
| bunch_562_jpg.rf.0305d361f069c58873086bb | 1 | 1 | 1 | 100% | 0 | 0.31 |
| bunch_570_jpg.rf.39623afff4f3c3861b3f4b8 | 1 | 4 | 1 | 100% | 3 | 0.55 |
| bunch_575_jpg.rf.cb0225d1f2d39866d773330 | 1 | 7 | 1 | 100% | 6 | 0.56 |
| bunch_579_jpg.rf.5407cbad3dfc41244dd7ec6 | 1 | 2 | 1 | 100% | 1 | 0.44 |
| bunch_593_jpg.rf.facfe498c83392073b84c56 | 1 | 6 | 1 | 100% | 5 | 0.85 |
| bunch_611_jpg.rf.131d60f40ec7f35650e50d8 | 1 | 7 | 1 | 100% | 6 | 1.53 |
| bunch_78_jpg.rf.555d926aa73ea27af6c37eb7 | 1 | 7 | 1 | 100% | 6 | 0.73 |
| bunch_79_jpg.rf.4e14a318d3f4aea383f12b52 | 2 | 4 | 2 | 100% | 2 | 0.68 |
| bunch_82_jpg.rf.dea869305eddf28a1e4552c2 | 2 | 4 | 2 | 100% | 2 | 0.61 |
| bunch_92_jpg.rf.a004bc12cf3c0610cbd6e9a5 | 2 | 7 | 2 | 100% | 5 | 0.63 |
| bunch_97_jpg.rf.0ee9408bb835d780c8faff63 | 6 | 8 | 2 | 33% | 6 | 1.02 |
