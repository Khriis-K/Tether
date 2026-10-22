# spec-verify: issue #43, fine-tuning the drone detector

Spec: the acceptance criteria in [issue #43](https://github.com/Khriis-K/Tether/issues/43) and the training choices of [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md). Built as `tether.drone_dataset` (the training set) and `tools/train_drone.py` (the settings and the export).
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Tests are in `python/tests/test_drone_dataset.py` and `test_train_drone.py`. 16 of 17 mutations were caught. The first run caught 15: the absent-share test compared the sampled share with a source share computed by the code under test, so a mutant that made them the same thing passed. The test now computes the source share from the labels itself. The one survivor is an equivalent mutant (C2).

### Criterion C1: no new-val or test session appears in the training set
Test: test_the_training_images_come_only_from_the_splits_train_sessions, test_val_images_come_only_from_the_splits_val_sessions_and_test_is_never_read (test's files are deleted first, so any read of them fails)
Mutation: test's clips are read too.
Result: VERIFIED
Mutation: train and val clips are swapped.
Result: VERIFIED

### Criterion C2: the stride is configurable and applied from the first frame
Test: test_every_stride_th_frame_is_sampled_from_the_first, test_a_stride_below_one_is_refused (0 and -1), test_each_clip_gives_its_stride_th_frames_with_a_label_file_each, test_the_cli_writes_the_settings_beside_the_dataset (stride 5)
Mutation: sampling starts at frame 1.
Result: VERIFIED
Mutation: the guard becomes `stride < 0`, so 0 gets past it.
Result: SURVIVED, an equivalent mutant: `range()` refuses a step of 0 by itself, and -1 is still refused by the guard.

### Criterion C3: the drone-absent share matches the source within a small tolerance
Test: test_the_drone_absent_share_matches_the_source (2,000-frame clips, 5% absent at random; within 0.02 at stride 10, with the source share checked against the labels)
Mutation: drone-absent frames are skipped.
Result: VERIFIED
Mutation: the source share is computed from the sampled images instead of every frame.
Result: VERIFIED (after the test fix above; SURVIVED before it)

### Criterion C4: labels are the frame's drone box in YOLO format, one class
Test: test_a_yolo_label_is_the_box_center_and_size_over_the_frame_size, test_an_absent_drone_has_an_empty_label, test_a_box_over_the_frame_edge_is_clipped_to_the_frame, test_a_box_wholly_outside_the_frame_has_an_empty_label, test_an_images_label_is_its_frames_box, test_images_are_resized_to_the_given_width_keeping_the_aspect_ratio, test_the_dataset_file_names_one_class_drone_and_the_two_image_folders
Mutation: the box isn't clipped to the frame.
Result: VERIFIED
Mutation: a box wholly outside the frame is kept.
Result: VERIFIED
Mutation: x is normalized by the height.
Result: VERIFIED
Mutation: images are resized without keeping the aspect ratio.
Result: VERIFIED
Mutation: data.yaml names the class "item".
Result: VERIFIED

### Criterion C5: the run's settings (epochs, batch, seed, stride) are committed so it can be repeated
Test: test_training_is_seeded_and_deterministic_so_it_can_be_repeated, test_the_epochs_and_run_folder_are_the_given_ones, test_every_epochs_checkpoint_is_kept_so_it_can_be_chosen_on_val_ap50; the stride is in the dataset's `dataset.json`, and Ultralytics' full settings are in `docs/benchmarks/drone-model-train-args.yaml`
Mutation: the seed changes.
Result: VERIFIED
Mutation: deterministic mode is off.
Result: VERIFIED
Mutation: per-epoch checkpoints aren't kept.
Result: VERIFIED

### Criterion C6: the ONNX export carries its class name, and can't collide with the COCO model
Test: test_training_keeps_the_datasets_class_name, test_the_exported_class_is_drone, test_an_export_without_drone_in_its_name_is_refused, test_an_export_named_drone_is_accepted
Mutation: `single_cls` is turned back on (Ultralytics then names the class "item").
Result: VERIFIED
Mutation: the exported class is named "item".
Result: VERIFIED
Mutation: the export name isn't checked.
Result: VERIFIED
The exported file was also checked by hand: `models/yolo26n_drone_1280x736.onnx` has input [1, 3, 736, 1280], output [1, 300, 6] and metadata `names: {0: 'drone'}`.

### Not covered by a unit test
- Checkpoint choice by val AP50 and the threshold by max F1 on val are steps run by hand with `tether-drone-score` (#42's tested scorer); the record is `docs/benchmarks/drone-model.md` and `drone-model-candidates.json`.
- Training itself (Ultralytics) isn't unit-tested; its run is recorded in `drone-model-train-results.csv`.
