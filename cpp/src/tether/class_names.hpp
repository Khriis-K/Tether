// What to call each class ID on screen. A model exported by Ultralytics names
// its classes in its ONNX metadata ("names": "{0: 'drone'}"); a TensorRT engine
// carries no metadata, so the .onnx beside it is read, and --class-names covers
// an engine with none. COCO's names are the fallback, for models without any.
// Mirrors python/tether/class_names.py.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace tether {

using ClassNames = std::vector<std::string>;  // indexed by class ID

// The 80 COCO classes the pretrained YOLO26 models detect.
const ClassNames& coco_names();

// Ultralytics' "names" metadata, a Python dict literal: "{0: 'person', 1: 'bicycle'}".
// Throws std::invalid_argument unless it parses and its IDs are 0, 1, ... in order.
ClassNames parse_names_metadata(const std::string& text);

// --class-names: comma-separated names by class ID ("drone" or "drone,bird").
// Throws std::invalid_argument for an empty name.
ClassNames parse_names_flag(const std::string& text);

// The names in `model`'s ONNX metadata; empty when it has none, the file isn't
// there, or it isn't an .onnx file (a TensorRT engine). Throws
// std::runtime_error, naming the model, when they can't be read or parsed.
std::optional<ClassNames> model_class_names(const std::filesystem::path& model);

// The flag's names if given, else the model's, else COCO's.
ClassNames class_names_for(const std::filesystem::path& model, const std::optional<std::string>& flag);

}  // namespace tether
