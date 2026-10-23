#include "tether/class_names.hpp"

#include <cctype>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>

namespace tether {

const ClassNames& coco_names() {
  static const ClassNames names = {
      "person",        "bicycle",      "car",           "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",          "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",          "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",       "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket", "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",         "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",         "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",            "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",     "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",         "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
  return names;
}

ClassNames parse_names_metadata(const std::string& text) {
  auto bad = [&] { return std::invalid_argument("class names metadata '" + text + "' isn't {0: 'name', ...}"); };
  size_t i = 0;
  auto skip_space = [&] {
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
  };
  auto expect = [&](char c) {
    skip_space();
    if (i >= text.size() || text[i] != c) throw bad();
    ++i;
  };
  ClassNames names;
  expect('{');
  while (true) {
    skip_space();
    size_t digits = i;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) ++i;
    if (i == digits || text.substr(digits, i - digits) != std::to_string(names.size())) throw bad();
    expect(':');
    skip_space();
    if (i >= text.size() || (text[i] != '\'' && text[i] != '"')) throw bad();
    char quote = text[i++];
    size_t end = text.find(quote, i);
    if (end == std::string::npos) throw bad();
    names.push_back(text.substr(i, end - i));
    i = end + 1;
    skip_space();
    if (i >= text.size() || text[i] != ',') break;
    ++i;
  }
  expect('}');
  skip_space();
  if (i != text.size()) throw bad();
  return names;
}

ClassNames parse_names_flag(const std::string& text) {
  ClassNames names;
  size_t start = 0;
  while (true) {
    size_t comma = text.find(',', start);
    names.push_back(text.substr(start, comma - start));
    if (names.back().empty()) throw std::invalid_argument("--class-names has an empty name: '" + text + "'");
    if (comma == std::string::npos) return names;
    start = comma + 1;
  }
}

std::optional<ClassNames> model_class_names(const std::filesystem::path& model) {
  // An engine (--model X.engine) has no metadata, and ONNX Runtime can't open it.
  if (model.extension() != ".onnx" || !std::filesystem::exists(model)) return std::nullopt;
  try {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "tether"};
    Ort::Session session(env, model.c_str(), Ort::SessionOptions{});
    Ort::AllocatorWithDefaultOptions allocator;
    auto names = session.GetModelMetadata().LookupCustomMetadataMapAllocated("names", allocator);
    if (!names) return std::nullopt;
    return parse_names_metadata(names.get());
  } catch (const std::exception& e) {  // Ort::Exception or std::invalid_argument
    throw std::runtime_error(model.string() + ": " + e.what());
  }
}

ClassNames class_names_for(const std::filesystem::path& model, const std::optional<std::string>& flag) {
  if (flag) return parse_names_flag(*flag);
  return model_class_names(model).value_or(coco_names());
}

}  // namespace tether
