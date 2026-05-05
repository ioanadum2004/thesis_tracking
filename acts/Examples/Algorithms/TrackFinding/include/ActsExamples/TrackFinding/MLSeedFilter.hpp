// ReadDataHandle<TrackParametersContainer> m_inputTrackParameters{this, "InputTrackParameters"};
// WriteDataHandle<TrackParametersContainer> m_outputTrackParameters{this, "OutputTrackParameters"};
// ```

/// MLSeedFilter: an ACTS algorithm that applies a trained ML model to filter
/// seeds before the CKF step. Reads estimated track parameters and seeds from
/// the whiteboard, extracts 12/27 track parameter features, runs ONNX inference,
/// and writes back only seeds whose score exceeds a threshold.

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
  struct Config { /// what configurations are needed for this algorithm, e.g. input/output collection names, model path, scaler parameters, threshold
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
    /// change this if u change threshold
  };

  MLSeedFilter(Config cfg, Acts::Logging::Level lvl);                // constructor, initialize data handles and ONNX session

  ProcessCode execute(const AlgorithmContext& ctx) const final;      // main method, read seeds and track parameters, extract features, run inference, apply threshold, write output

  const Config& config() const { return m_cfg; }

 private:
  Config m_cfg;

  // ONNX Runtime session for inference
  Ort::Env m_env;
  mutable Ort::Session m_session;

  /// what data it reads and writes
  ReadDataHandle<TrackParametersContainer> m_inputTrackParameters{
      this, "InputTrackParameters"};  // track parameters from trackParameterContainer, needed for feature extraction
  WriteDataHandle<TrackParametersContainer> m_outputTrackParameters{
      this, "OutputTrackParameters"};
  ReadDataHandle<SimSeedContainer> m_inputSeeds{
      this, "InputSeeds"};            // seeds from simSeedContainer
  WriteDataHandle<SimSeedContainer> m_outputSeeds{
      this, "OutputSeeds"};
  
};

}  // namespace ActsExamples
