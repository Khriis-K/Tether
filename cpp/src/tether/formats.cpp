#include "tether/formats.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tether {

namespace {

// A double as Python's json.dumps writes it (Python's repr): the shortest digits
// that read back to the same value, in fixed notation for exponents -4 to 15 and
// scientific otherwise. Matching it keeps the two trackers' files alike.
std::string number(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  char buf[32];
  auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string_view sci(buf, end - buf);  // e.g. "-1.25e+03"
  auto e_pos = sci.find('e');
  int exponent = std::stoi(std::string(sci.substr(e_pos + 1)));
  if (exponent < -4 || exponent >= 16) return std::string(sci);  // already Python's form

  std::string_view mantissa = sci.substr(0, e_pos);
  std::string sign = mantissa.front() == '-' ? "-" : "";
  if (!sign.empty()) mantissa.remove_prefix(1);
  std::string digits;
  for (char c : mantissa)
    if (c != '.') digits += c;

  if (exponent < 0) return sign + "0." + std::string(-exponent - 1, '0') + digits;
  auto int_len = static_cast<size_t>(exponent) + 1;
  if (digits.size() <= int_len) return sign + digits + std::string(int_len - digits.size(), '0') + ".0";
  return sign + digits.substr(0, int_len) + "." + digits.substr(int_len);
}

template <size_t N>
std::string numbers(const std::array<double, N>& values) {
  std::string out = "[";
  for (size_t i = 0; i < N; ++i) out += (i ? ", " : "") + number(values[i]);
  return out + "]";
}

std::string detection_json(const Detection& d) {
  return "{\"box\": " + numbers(d.box) + ", \"score\": " + number(d.score) +
         ", \"class\": " + std::to_string(d.cls) + "}";
}

std::string track_json(const Track& t) {
  std::string path = "[";
  for (size_t i = 0; i < t.path.size(); ++i) path += (i ? ", " : "") + numbers(t.path[i]);
  path += "]";
  // Status is one of three fixed words, so it needs no escaping.
  return "{\"id\": " + std::to_string(t.id) + ", \"box\": " + numbers(t.box) +
         ", \"class\": " + std::to_string(t.cls) + ", \"status\": \"" + t.status +
         "\", \"predicted_box\": " + numbers(t.predicted_box) +
         ", \"frames_since_seen\": " + std::to_string(t.frames_since_seen) + ", \"path\": " + path + "}";
}

// One line per frame: {"frame": i, "<key>": [items...]}.
template <typename T, typename ToJson>
void write_lines(const std::filesystem::path& path, const std::vector<std::vector<T>>& frames,
                 const char* key, ToJson to_json) {
  // Binary, so "\n" stays "\n" on Windows and the bytes match Linux.
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(path.string() + ": can't open for writing");
  for (size_t i = 0; i < frames.size(); ++i) {
    f << "{\"frame\": " << i << ", \"" << key << "\": [";
    for (size_t j = 0; j < frames[i].size(); ++j) f << (j ? ", " : "") << to_json(frames[i][j]);
    f << "]}\n";
  }
  if (!f) throw std::runtime_error(path.string() + ": write failed");
}

// Runs `read`, turning a JSON error (a missing field, a wrong type) into std::runtime_error.
template <typename Read>
auto rethrowing(const std::filesystem::path& path, Read read) {
  try {
    return read();
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(path.string() + ": " + e.what());
  }
}

// The `key` array of every non-blank line, checking that line i is frame i.
std::vector<nlohmann::json> read_lines(const std::filesystem::path& path, const char* key) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(path.string() + ": can't open for reading");
  std::vector<nlohmann::json> frames;
  std::string line;
  while (std::getline(f, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    auto obj = rethrowing(path, [&] { return nlohmann::json::parse(line); });
    auto frame = rethrowing(path, [&] { return obj.at("frame").get<long long>(); });
    if (frame != static_cast<long long>(frames.size()))
      throw std::runtime_error(path.string() + ": expected frame " + std::to_string(frames.size()) +
                               ", found frame " + std::to_string(frame));
    frames.push_back(rethrowing(path, [&] { return obj.at(key); }));
  }
  return frames;
}

template <size_t N>
std::array<double, N> to_array(const nlohmann::json& j) {
  if (j.size() != N) throw std::runtime_error("expected " + std::to_string(N) + " numbers, got " + j.dump());
  std::array<double, N> out;
  for (size_t i = 0; i < N; ++i) out[i] = j[i].get<double>();
  return out;
}

std::ofstream open_for_writing(const std::filesystem::path& path) {
  std::ofstream f(path, std::ios::binary);  // binary, so "\n" stays "\n" on Windows
  if (!f) throw std::runtime_error(path.string() + ": can't open for writing");
  return f;
}

void write_line(std::ofstream& f, const std::filesystem::path& path, const std::string& line) {
  f << line << '\n';
  f.flush();
  if (!f) throw std::runtime_error(path.string() + ": write failed");
}

}  // namespace

void save_detections(const std::filesystem::path& path, const DetectionFrames& frames) {
  DetectionWriter writer(path);
  for (const auto& dets : frames) writer.write(dets);
}

DetectionFrames load_detections(const std::filesystem::path& path) {
  DetectionFrames frames;
  for (const auto& line : read_lines(path, "detections")) {
    auto& dets = frames.emplace_back();
    for (const auto& d : line) dets.push_back(rethrowing(path, [&] {
      return Detection{to_array<4>(d.at("box")), d.at("score").get<double>(), d.at("class").get<int>()};
    }));
  }
  return frames;
}

void save_tracks(const std::filesystem::path& path, const TrackFrames& frames) {
  write_lines(path, frames, "tracks", track_json);
}

TrackFrames load_tracks(const std::filesystem::path& path) {
  TrackFrames frames;
  for (const auto& line : read_lines(path, "tracks")) {
    auto& tracks = frames.emplace_back();
    for (const auto& t : line) tracks.push_back(rethrowing(path, [&] {
      std::vector<Point> trail;
      for (const auto& p : t.at("path")) trail.push_back(to_array<2>(p));
      return Track{t.at("id").get<int>(), to_array<4>(t.at("box")), t.at("class").get<int>(),
                   t.at("status").get<std::string>(), to_array<4>(t.at("predicted_box")),
                   t.at("frames_since_seen").get<int>(), std::move(trail)};
    }));
  }
  return frames;
}

DetectionWriter::DetectionWriter(const std::filesystem::path& path) : path_(path), file_(open_for_writing(path)) {}

void DetectionWriter::write(const std::vector<Detection>& detections) {
  std::string line = "{\"frame\": " + std::to_string(frame_++) + ", \"detections\": [";
  for (size_t j = 0; j < detections.size(); ++j) line += (j ? ", " : "") + detection_json(detections[j]);
  write_line(file_, path_, line + "]}");
}

TimingLogWriter::TimingLogWriter(const std::filesystem::path& path) : path_(path), file_(open_for_writing(path)) {}

void TimingLogWriter::write(const StepTimes& t) {
  std::string line = "{\"frame\": " + std::to_string(frame_++) + ", \"steps\": {\"capture\": " +
                     numbers(t.capture) + ", \"prep\": " + numbers(t.prep) + ", \"detect\": " +
                     numbers(t.detect) + ", \"track\": " + numbers(t.track) + ", \"draw\": " +
                     numbers(t.draw) + "}";
  if (t.dropped) line += ", \"dropped\": " + std::to_string(*t.dropped);
  write_line(file_, path_, line + "}");
}

}  // namespace tether
