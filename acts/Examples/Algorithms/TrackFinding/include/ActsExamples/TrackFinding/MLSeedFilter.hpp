// ReadDataHandle<TrackParametersContainer> m_inputTrackParameters{this, "InputTrackParameters"};
// WriteDataHandle<TrackParametersContainer> m_outputTrackParameters{this, "OutputTrackParameters"};
// ```

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace ActsExamples {

struct AlgorithmContext;

/// explicatii gen ce face tot algoritmul, un soi de pseudocod pe care o sa l pun mai tz
class MLSeedFilter final : public IAlgorithm {
 public:
  struct Config {
    /// Input track parameters collection
    std::string inputTrackParameters;
    /// Output filtered track parameters collection
    std::string outputTrackParameters;
    ///Input seeds collection
    std::string inputSeeds;
    ///Output seeds collection
    std::string outputSeeds;
    /// Path to the ONNX model file
    std::string modelPath;
    /// Scaler means (one per feature)
    std::vector<float> scalerMeans;
    /// Scaler standard deviations (one per feature)
    std::vector<float> scalerStds;
    /// Threshold for keeping a seed (0.0 - 1.0)
    float threshold = 0.4f;
  };

  MLSeedFilter(Config cfg, Acts::Logging::Level lvl);

  ProcessCode execute(const AlgorithmContext& ctx) const final;

  const Config& config() const { return m_cfg; }

 private:
  Config m_cfg;

  // ONNX Runtime session for inference
  Ort::Env m_env;
  mutable Ort::Session m_session;

  ReadDataHandle<TrackParametersContainer> m_inputTrackParameters{
      this, "InputTrackParameters"};
  WriteDataHandle<TrackParametersContainer> m_outputTrackParameters{
      this, "OutputTrackParameters"};
  ReadDataHandle<SimSeedContainer> m_inputSeeds{this, "InputSeeds"};
  WriteDataHandle<SimSeedContainer> m_outputSeeds{this, "OutputSeeds"};
  
};

}  // namespace ActsExamples
