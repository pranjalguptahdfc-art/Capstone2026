# Blind-batch report — live pipeline vs. Training labels

Batch size: 20 (limit 20)  ·  splits: valid + test  ·  detection: adaptive auto mode (conf 0.1, floor off)  ·  match IoU ≥ 0.3  ·  generated 2026-10-28 12:48

## Aggregates

| metric | value |
|---|---|
| images | 20 ok / 0 errored |
| mean detection recall (images with labels) | 66% over 19 |
| images with zero detections | 0 |
| matched bananas | 26 |
| ripeness agreement | 11 / 26 (42%) |
| spurious detections (no label overlap) | 63 |

## Confusion (label row × predicted col)

| label \ pred | overripe | ripe | semi-ripe | underripe | unlabelled |
|---|---|---|---|---|---|
| overripe | 1 | 5 | 0 | 0 | 0 |
| ripe | 0 | 5 | 0 | 1 | 0 |
| underripe | 0 | 9 | 2 | 3 | 0 |
| unlabelled | 0 | 0 | 0 | 0 | 0 |

## Mismatches (matched banana whose class disagrees)

| image | # | label | pred | G% | Y% | B% | det conf |
|---|---|---|---|---|---|---|---|
| bunch_643_jpg.rf.67f35b648ec4d | 4 | overripe | ripe | 0.1 | 81.9 | 16.4 | 0.32 |
| bunch_643_jpg.rf.67f35b648ec4d | 1 | overripe | ripe | 0.0 | 55.7 | 0.4 | 0.40 |
| bunch_647_jpg.rf.e9067b006cf5c | 1 | overripe | ripe | 0.0 | 34.8 | 1.0 | 0.21 |
| bunch_647_jpg.rf.e9067b006cf5c | 2 | overripe | ripe | 0.0 | 7.4 | 0.0 | 0.16 |
| bunch_440_jpg.rf.9ae90b20e93d4 | 1 | overripe | ripe | 0.1 | 52.0 | 7.1 | 0.18 |
| bunch_575_jpg.rf.cb0225d1f2d39 | 1 | underripe | ripe | 6.3 | 81.8 | 7.4 | 0.28 |
| bunch_185_jpg.rf.b621f38f16a29 | 1 | underripe | ripe | 0.2 | 86.6 | 4.9 | 0.35 |
| bunch_45_jpg.rf.ded71a67a7b9ff | 1 | underripe | ripe | 0.5 | 97.1 | 2.3 | 0.61 |
| bunch_291_jpg.rf.e5dca3811156e | 2 | underripe | ripe | 0.2 | 97.2 | 1.9 | 0.19 |
| bunch_291_jpg.rf.e5dca3811156e | 1 | ripe | unripe | 0.0 | 0.0 | 0.0 | 0.47 |
| bunch_202_jpg.rf.bbcd851d25c8d | 1 | underripe | ripe | 0.0 | 87.4 | 11.6 | 0.41 |
| bunch_559_jpg.rf.c4ba31e84dedf | 1 | underripe | ripe | 15.2 | 79.6 | 2.6 | 0.39 |
| bunch_42_jpg.rf.517a873a24fbb0 | 6 | underripe | ripe | 16.9 | 72.5 | 1.4 | 0.66 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 1 | underripe | ripe | 0.2 | 86.9 | 4.9 | 0.56 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 2 | underripe | ripe | 0.1 | 96.8 | 3.0 | 0.76 |

## Per-image detail

| image | labels | dets | matched | recall | spurious | sec |
|---|---|---|---|---|---|---|
| bunch_100_jpg.rf.a1b2603b38385cde1eb1782 | 1 | 2 | 0 | 0% | 2 | 4.9 |
| bunch_643_jpg.rf.67f35b648ec4dfd74dbfae1 | 6 | 3 | 2 | 33% | 1 | 1.13 |
| bunch_647_jpg.rf.e9067b006cf5ca75b95818c | 4 | 10 | 3 | 75% | 7 | 0.95 |
| bunch_426_jpg.rf.63d4554df0df892a4a1b968 | 1 | 1 | 1 | 100% | 0 | 0.8 |
| bunch_440_jpg.rf.9ae90b20e93d48a5599b6aa | 1 | 1 | 1 | 100% | 0 | 0.31 |
| bunch_618_jpg.rf.3dc1311a5cf991afaa76c5d | 1 | 4 | 0 | 0% | 4 | 0.69 |
| bunch_84_jpg.rf.2a87ae6c1e86623b3ba1ca67 | 1 | 3 | 0 | 0% | 3 | 0.29 |
| bunch_575_jpg.rf.cb0225d1f2d39866d773330 | 1 | 7 | 1 | 100% | 6 | 0.68 |
| bunch_432_jpg.rf.5a0099d014b0321991e3188 | 1 | 8 | 1 | 100% | 7 | 1.87 |
| bunch_185_jpg.rf.b621f38f16a295aebbd47e2 | 3 | 9 | 2 | 67% | 7 | 0.62 |
| bunch_45_jpg.rf.ded71a67a7b9ff84df88a891 | 4 | 7 | 3 | 75% | 4 | 0.82 |
| bunch_101_jpg.rf.d1a1e7823b929082e9a9f4f | 3 | 1 | 1 | 33% | 0 | 1.02 |
| bunch_291_jpg.rf.e5dca3811156e90426e85a1 | 2 | 11 | 2 | 100% | 9 | 0.91 |
| bunch_517_jpg.rf.87bb8c04473353edd7f9ec4 | 1 | 2 | 1 | 100% | 1 | 0.93 |
| bunch_202_jpg.rf.bbcd851d25c8d78fed28b3f | 2 | 5 | 2 | 100% | 3 | 0.52 |
| bunch_559_jpg.rf.c4ba31e84dedf364ee7441d | 1 | 6 | 1 | 100% | 5 | 0.76 |
| bunch_42_jpg.rf.517a873a24fbb0d2c9d1298b | 7 | 1 | 1 | 14% | 0 | 0.21 |
| null_33_jpg.rf.96935df24891f67395b37f95d | 0 | 2 | 0 | — | 2 | 0.18 |
| bunch_633_jpeg.rf.c0d7e698897ba31c9d3209 | 4 | 2 | 2 | 50% | 0 | 0.32 |
| bunch_79_jpg.rf.4e14a318d3f4aea383f12b52 | 2 | 4 | 2 | 100% | 2 | 0.72 |
