// The detector: decoding YOLO26 output, and the ONNX Runtime detector on stand-in
// models (tests/onnx/make_models.py). Mirrors python/tests/test_detector.py and test_input_size.py.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

#include "test_util.hpp"
#include "tether/detector.hpp"

using namespace tether;
using tether::test::boxes_near;

namespace {

const std::string DATA = TETHER_TEST_DATA;

PreparedFrame geometry(double scale, int pad_x, int pad_y, int frame_h, int frame_w) {
  return {{}, 0, 0, scale, pad_x, pad_y, frame_h, frame_w};
}

}  // namespace

TEST(Decode, MapsBoxesBackToFramePixels) {
  // Frame 1280x720 letterboxed to 640: scale 0.5, 140 px of padding on top.
  auto dets = decode({100, 190, 200, 290, 0.9f, 0}, geometry(0.5, 0, 140, 720, 1280), 0.1);

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].box, (Box{200.0, 100.0, 400.0, 300.0}));
  EXPECT_EQ(dets[0].score, static_cast<double>(0.9f));
  EXPECT_EQ(dets[0].cls, 0);
}

TEST(Decode, MapsBoxesFromARectangularInputBackToFramePixels) {
  // 1920x1080 frame letterboxed to 1280x736: scale 2/3, 8 px of padding on top.
  auto dets = decode({100, 108, 300, 308, 0.9f, 0}, geometry(2.0 / 3.0, 0, 8, 1080, 1920), 0.1);

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_TRUE(boxes_near(dets[0].box, {150.0, 150.0, 450.0, 450.0}, 1e-3));
}

TEST(Decode, KeepsScoresAtTheThresholdAndDropsOnesBelow) {
  auto dets = decode({10, 10, 20, 20, 0.1f, 0,  //
                      10, 10, 20, 20, 0.09f, 0,  //
                      30, 30, 40, 40, 0.5f, 2},
                     geometry(1.0, 0, 0, 100, 100), 0.1);

  ASSERT_EQ(dets.size(), 2u);
  EXPECT_EQ(dets[0].score, static_cast<double>(0.1f));
  EXPECT_EQ(dets[1].cls, 2);
}

TEST(Decode, ClipsBoxesToTheFrameAndDropsOnesLeftEmpty) {
  auto dets = decode({-5, -5, 50, 120, 0.9f, 0,  // pokes out top-left and bottom
                      150, 10, 160, 20, 0.9f, 0},  // entirely right of the frame
                     geometry(1.0, 0, 0, 100, 100), 0.1);

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].box, (Box{0.0, 0.0, 50.0, 100.0}));
}

TEST(Decode, DoesTheMathInFloat32AsNumpyDoes) {
  // Python's decode gives 337.6666564941406 here (float32); double math would give 337.66667683919275.
  auto dets = decode({101.3f, 0, 111.3f, 10, 0.9f, 0}, geometry(0.3, 0, 0, 1080, 1920), 0.1);

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].box[0], 337.6666564941406);
}

TEST(ModelInputSize, ReadsHeightAndWidthFromAnNchwShape) {
  EXPECT_EQ(model_input_size({1, 3, 736, 1280}), (std::pair<int, int>{736, 1280}));
}

TEST(ModelInputSize, RejectsDynamicOrMalformedShapes) {
  EXPECT_THROW(model_input_size({1, 3, -1, -1}), std::invalid_argument);
  EXPECT_THROW(model_input_size({-1, 3, 640, 640}), std::invalid_argument);
  EXPECT_THROW(model_input_size({1, 3, 640}), std::invalid_argument);
}

TEST(OnnxDetector, ReadsItsInputSizeFromTheModel) {
  OnnxDetector detector(DATA + "/fixed_736x1280.onnx");

  EXPECT_EQ(detector.input_size(), (std::pair<int, int>{736, 1280}));
}

TEST(OnnxDetector, MapsBoxesFromARectangularModelBackToTheFrame) {
  OnnxDetector detector(DATA + "/fixed_736x1280.onnx");
  cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));

  auto dets = detector.detect(CpuFramePrep(736, 1280).prepare(frame));

  ASSERT_EQ(dets.size(), 1u);
  EXPECT_TRUE(boxes_near(dets[0].box, {150.0, 150.0, 450.0, 450.0}, 1e-3));
  EXPECT_EQ(dets[0].score, static_cast<double>(0.9f));
}

TEST(OnnxDetector, RejectsAModelWithADynamicInputSize) {
  try {
    OnnxDetector detector(DATA + "/dynamic_input.onnx");
    FAIL() << "expected an error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("fixed"), std::string::npos) << e.what();
  }
}

TEST(OnnxDetector, AMissingModelIsAnError) {
  EXPECT_THROW(OnnxDetector(DATA + "/no-such-model.onnx"), std::runtime_error);
}
