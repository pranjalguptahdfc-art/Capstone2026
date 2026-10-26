# Blind-batch report — live pipeline vs. Training labels

Batch size: 20 (limit 20)  ·  splits: valid + test  ·  detection: adaptive auto mode (conf 0.1, floor off)  ·  match IoU ≥ 0.3  ·  generated 2026-10-26 18:16

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
| overripe | 6 | 0 | 0 | 0 | 0 |
| ripe | 6 | 0 | 0 | 0 | 0 |
| underripe | 6 | 3 | 1 | 4 | 0 |
| unlabelled | 0 | 0 | 0 | 0 | 0 |

## Mismatches (matched banana whose class disagrees)

| image | # | label | pred | G% | Y% | B% | det conf |
|---|---|---|---|---|---|---|---|
| bunch_647_jpg.rf.e9067b006cf5c | 3 | ripe | overripe | 0.0 | 52.6 | 41.8 | 0.15 |
| bunch_575_jpg.rf.cb0225d1f2d39 | 1 | underripe | overripe | 5.3 | 62.0 | 56.8 | 0.28 |
| bunch_432_jpg.rf.5a0099d014b03 | 1 | ripe | overripe | 0.0 | 58.7 | 81.9 | 0.39 |
| bunch_185_jpg.rf.b621f38f16a29 | 2 | ripe | overripe | 0.0 | 62.5 | 76.6 | 0.34 |
| bunch_185_jpg.rf.b621f38f16a29 | 1 | underripe | overripe | 0.0 | 98.8 | 50.5 | 0.35 |
| bunch_45_jpg.rf.ded71a67a7b9ff | 1 | underripe | overripe | 3.9 | 99.0 | 24.2 | 0.61 |
| bunch_101_jpg.rf.d1a1e7823b929 | 1 | ripe | overripe | 0.0 | 29.5 | 99.0 | 0.25 |
| bunch_291_jpg.rf.e5dca3811156e | 2 | underripe | overripe | 1.2 | 98.9 | 62.2 | 0.19 |
| bunch_291_jpg.rf.e5dca3811156e | 1 | ripe | overripe | 0.0 | 0.0 | 8.3 | 0.47 |
| bunch_202_jpg.rf.bbcd851d25c8d | 2 | ripe | overripe | 0.0 | 71.2 | 78.8 | 0.27 |
| bunch_202_jpg.rf.bbcd851d25c8d | 1 | underripe | overripe | 0.0 | 78.5 | 82.4 | 0.41 |
| bunch_559_jpg.rf.c4ba31e84dedf | 1 | underripe | ripe | 15.1 | 81.7 | 18.3 | 0.39 |
| bunch_42_jpg.rf.517a873a24fbb0 | 6 | underripe | ripe | 16.6 | 55.9 | 5.1 | 0.66 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 1 | underripe | overripe | 0.2 | 78.8 | 50.6 | 0.56 |
| bunch_79_jpg.rf.4e14a318d3f4ae | 2 | underripe | ripe | 3.3 | 99.6 | 16.9 | 0.76 |

## Per-image detail

| image | labels | dets | matched | recall | spurious | sec |
|---|---|---|---|---|---|---|
| bunch_100_jpg.rf.a1b2603b38385cde1eb1782 | 1 | 2 | 0 | 0% | 2 | 3.69 |
| bunch_643_jpg.rf.67f35b648ec4dfd74dbfae1 | 6 | 3 | 2 | 33% | 1 | 0.78 |
| bunch_647_jpg.rf.e9067b006cf5ca75b95818c | 4 | 10 | 3 | 75% | 7 | 0.9 |
| bunch_426_jpg.rf.63d4554df0df892a4a1b968 | 1 | 1 | 1 | 100% | 0 | 0.59 |
| bunch_440_jpg.rf.9ae90b20e93d48a5599b6aa | 1 | 1 | 1 | 100% | 0 | 0.22 |
| bunch_618_jpg.rf.3dc1311a5cf991afaa76c5d | 1 | 4 | 0 | 0% | 4 | 0.47 |
| bunch_84_jpg.rf.2a87ae6c1e86623b3ba1ca67 | 1 | 3 | 0 | 0% | 3 | 0.22 |
| bunch_575_jpg.rf.cb0225d1f2d39866d773330 | 1 | 7 | 1 | 100% | 6 | 0.51 |
| bunch_432_jpg.rf.5a0099d014b0321991e3188 | 1 | 8 | 1 | 100% | 7 | 1.57 |
| bunch_185_jpg.rf.b621f38f16a295aebbd47e2 | 3 | 9 | 2 | 67% | 7 | 0.52 |
| bunch_45_jpg.rf.ded71a67a7b9ff84df88a891 | 4 | 7 | 3 | 75% | 4 | 0.2 |
| bunch_101_jpg.rf.d1a1e7823b929082e9a9f4f | 3 | 1 | 1 | 33% | 0 | 0.61 |
| bunch_291_jpg.rf.e5dca3811156e90426e85a1 | 2 | 11 | 2 | 100% | 9 | 0.63 |
| bunch_517_jpg.rf.87bb8c04473353edd7f9ec4 | 1 | 2 | 1 | 100% | 1 | 0.66 |
| bunch_202_jpg.rf.bbcd851d25c8d78fed28b3f | 2 | 5 | 2 | 100% | 3 | 0.34 |
| bunch_559_jpg.rf.c4ba31e84dedf364ee7441d | 1 | 6 | 1 | 100% | 5 | 0.56 |
| bunch_42_jpg.rf.517a873a24fbb0d2c9d1298b | 7 | 1 | 1 | 14% | 0 | 0.13 |
| null_33_jpg.rf.96935df24891f67395b37f95d | 0 | 2 | 0 | — | 2 | 0.11 |
| bunch_633_jpeg.rf.c0d7e698897ba31c9d3209 | 4 | 2 | 2 | 50% | 0 | 0.17 |
| bunch_79_jpg.rf.4e14a318d3f4aea383f12b52 | 2 | 4 | 2 | 100% | 2 | 0.55 |
